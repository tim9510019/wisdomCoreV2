import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ==========================================
# 0. SOTA 環境配置 (必須在 import torch 之前)
# ==========================================
os.environ["CUDA_VISIBLE_DEVICES"] = "1" # 強制鎖定第二張顯卡 (CUDA:1)
os.environ["BITSANDBYTES_NOWELCOME"] = "1"
# 防禦 96GB VRAM 碎片化，解決 23GB 碎片浪費問題
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, TrainerCallback, set_seed
from datasets import load_dataset, interleave_datasets
import math
import csv
import time

# 引入 AGIV2L.py (請確保該檔案已更新為 SDPA 版本)
from AGIV2L import AGIV2L

# 固定隨機種子確保隔離邏輯可重現
set_seed(42)

# ==========================================
# 1. 訓練包裝器 (動態 GC & Chunked Loss)
# ==========================================

class AGIV2LForCausalLM(nn.Module):
    def __init__(self, base_model, use_gc=True):
        super().__init__()
        self.base_model = base_model
        self.use_gc = use_gc

    def forward(self, input_ids, labels=None, **kwargs):
        hidden_states = self.base_model.embedding(input_ids)
        
        # 逐層前向傳播 (支援 Gradient Checkpointing)
        for i, block in enumerate(self.base_model.blocks):
            shift_size = (block.C // 2) if (i % 2 != 0) else 0
            if self.training and self.use_gc:
                # 使用最新官方推薦的 GC 呼叫方式
                hidden_states = checkpoint.checkpoint(
                    block, hidden_states, shift_size, use_reentrant=False
                )
            else:
                hidden_states = block(hidden_states, shift_size=shift_size)
                
        hidden_states = self.base_model.final_norm(hidden_states)
        
        loss = None
        logits = None
        
        if labels is not None:
            # 🚀 SOTA: Chunked Cross-Entropy (解決 15.6GB Logits 記憶體爆炸)
            shift_hidden = hidden_states[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss_fct = nn.CrossEntropyLoss(reduction="sum")
            total_loss = 0.0
            chunk_size = 2048 
            seq_len = shift_hidden.size(1)
            
            for i in range(0, seq_len, chunk_size):
                end_idx = min(i + chunk_size, seq_len)
                c_hidden = shift_hidden[:, i:end_idx, :]
                c_logits = self.base_model.fc_out(c_hidden)
                c_labels = shift_labels[:, i:end_idx]
                
                # 強制在 fp32 計算 Loss 確保數值精準
                c_loss = loss_fct(c_logits.reshape(-1, c_logits.size(-1)).float(), c_labels.reshape(-1))
                total_loss += c_loss
                
                # 手動釋放暫存 Tensor
                del c_logits, c_hidden
                
            loss = total_loss / shift_labels.numel()
        else:
            # 推論模式下才一次性生成 logits
            logits = self.base_model.fc_out(hidden_states)
            
        # 訓練中不回傳 logits 給 Trainer 以節省 VRAM
        return {"loss": loss, "logits": logits} if logits is not None else {"loss": loss}

# ==========================================
# 2. 權重移植、凍結與監控 Callback
# ==========================================

def transplant_and_freeze(model_id, agiv2_base):
    print(f"🔄 從 {model_id} 移植知識庫 (使用 dtype=bf16)...")
    # 使用最新語法 dtype 取代 torch_dtype
    src_model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        dtype=torch.bfloat16, 
        device_map="cpu"
    )
    src_sd = src_model.state_dict()
    new_sd = {}

    # 基礎權重映射 (Embedding / Head / FinalNorm)
    new_sd['embedding.weight'] = src_sd.get('model.embed_tokens.weight')
    new_sd['fc_out.weight'] = src_sd.get('lm_head.weight')
    new_sd['final_norm.weight'] = src_sd.get('model.norm.weight')

    # 32層權重映射
    for i in range(32):
        s_pre = f"model.layers.{i}."
        a_pre = f"blocks.{i}."
        is_global = (i + 1) % 4 == 0
        
        if is_global:
            new_sd[f"{a_pre}norm2.weight"] = src_sd.get(f"{s_pre}input_layernorm.weight")
            new_sd[f"{a_pre}norm3.weight"] = src_sd.get(f"{s_pre}post_attention_layernorm.weight")
        else:
            new_sd[f"{a_pre}norm1.weight"] = src_sd.get(f"{s_pre}input_layernorm.weight")
            new_sd[f"{a_pre}norm2.weight"] = src_sd.get(f"{s_pre}post_attention_layernorm.weight")
            
        new_sd[f"{a_pre}ffn.gate_proj.weight"] = src_sd.get(f"{s_pre}mlp.gate_proj.weight")
        new_sd[f"{a_pre}ffn.up_proj.weight"] = src_sd.get(f"{s_pre}mlp.up_proj.weight")
        new_sd[f"{a_pre}ffn.down_proj.weight"] = src_sd.get(f"{s_pre}mlp.down_proj.weight")

    # 載入權重到 AGIV2L 實體
    new_sd = {k: v for k, v in new_sd.items() if v is not None}
    agiv2_base.load_state_dict(new_sd, strict=False)
    
    # 執行物理凍結
    for name, param in agiv2_base.named_parameters():
        param.requires_grad = name not in new_sd
        if param.requires_grad:
            param.data = param.data.to(torch.bfloat16)

    del src_model
    import gc; gc.collect(); torch.cuda.empty_cache()
    return agiv2_base

class AGIV2LMonitor(TrainerCallback):
    """
    負責即時紀錄 Loss 到 CSV
    """
    def __init__(self, path="./agiv2_cpt_log.csv"):
        self.path = path
        with open(self.path, 'w', newline='') as f:
            csv.writer(f).writerow(['step', 'train_loss', 'eval_loss', 'lr', 'time'])

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            with open(self.path, 'a', newline='') as f:
                csv.writer(f).writerow([
                    state.global_step, 
                    logs.get("loss", ""), 
                    logs.get("eval_loss", ""), 
                    logs.get("learning_rate", ""), 
                    time.ctime()
                ])

# ==========================================
# 3. 資料集隔離 (Train: 前段, Val: 後段)
# ==========================================

class Packed32KDataset(torch.utils.data.IterableDataset):
    def __init__(self, tokenizer, max_length=32768, is_val=False):
        # 使用 100% 原生 Parquet 格式，避免腳本錯誤
        ds_edu = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
        ds_cos = load_dataset("HuggingFaceTB/cosmopedia", name="stanford", split="train", streaming=True)
        
        # 🚀 物理隔離邏輯：驗證集跳過前段資料
        if is_val:
            ds_edu, ds_cos = ds_edu.skip(100000), ds_cos.skip(10000)
        else:
            ds_edu, ds_cos = ds_edu.take(100000), ds_cos.take(10000)

        self.dataset = interleave_datasets([ds_edu, ds_cos], probabilities=[0.6, 0.4])
        self.tokenizer, self.max_length = tokenizer, max_length

    def __iter__(self):
        buffer = []
        for ex in self.dataset:
            text = ex.get('text','')
            if not text: continue
            tokens = self.tokenizer(text, add_special_tokens=False)['input_ids'] + [self.tokenizer.eos_token_id]
            buffer.extend(tokens)
            while len(buffer) >= self.max_length:
                chunk = buffer[:self.max_length]
                yield {
                    "input_ids": torch.tensor(chunk, dtype=torch.long), 
                    "labels": torch.tensor(chunk, dtype=torch.long)
                }
                buffer = buffer[self.max_length:]

# ==========================================
# 4. 主流程
# ==========================================

def main():
    # 使用免審核的平替模型 ID
    model_id = "NousResearch/Meta-Llama-3-8B"
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    # 1. 建立 AGIV2L 基礎架構 (1:3 稀疏配置)
    base = AGIV2L(vocab_size=len(tokenizer), D=4096, hidden_dim=14336, num_blocks=32)
    
    # 2. 移植權重並凍結
    base = transplant_and_freeze(model_id, base)
    
    # 3. 包裝模型
    model = AGIV2LForCausalLM(base).to(torch.bfloat16).cuda()

    # 4. 準備資料
    train_ds = Packed32KDataset(tokenizer, max_length=32768, is_val=False)
    val_ds = Packed32KDataset(tokenizer, max_length=4096, is_val=True) 

    # 5. 設定訓練參數 (採用最新 eval_strategy 語法)
    args = TrainingArguments(
        output_dir="./agiv2-cpt",
        max_steps=5000, 
        per_device_train_batch_size=1, 
        gradient_accumulation_steps=8,
        learning_rate=1e-4, 
        warmup_ratio=0.05, 
        lr_scheduler_type="cosine",
        weight_decay=0.01, 
        bf16=True, 
        logging_steps=10, 
        eval_strategy="steps", # ✅ 對齊最新版 transformers 語法
        eval_steps=100,
        save_steps=500, 
        optim="paged_adamw_8bit", 
        report_to="none", 
        gradient_checkpointing=False
    )

    # 6. 啟動 Trainer
    trainer = Trainer(
        model=model, 
        args=args, 
        train_dataset=train_ds, 
        eval_dataset=val_ds, 
        callbacks=[AGIV2LMonitor()]
    )
    
    print("🚀 開始 32K CPT 訓練 (CUDA:1)...")
    trainer.train()
    
    # 儲存結果
    torch.save(model.state_dict(), "./agiv2-cpt/final.pth")
    print("🎉 訓練完成！")

if __name__ == "__main__":
    main()