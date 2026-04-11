import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ==========================================
# 0. 硬體環境隔離 (必須在 import torch 之前設定)
# ==========================================
# 強制指定只使用硬體上的 GPU 1 (CUDA:1)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["BITSANDBYTES_NOWELCOME"] = "1"
# [新增這行] 允許 PyTorch 動態擴展記憶體段，完美解決 23GB 的碎片化浪費
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import load_dataset, interleave_datasets
import math

# 引入你的 SOTA 模型架構
from AGIV2L import AGIV2L

# ==========================================
# 1. 訓練包裝器 (動態梯度檢查點 & Loss 計算)
# ==========================================

# ==========================================
# 1. 訓練包裝器 (動態梯度檢查點 & 切塊 Loss 計算)
# ==========================================

class AGIV2LForCausalLM(nn.Module):
    def __init__(self, base_model, use_gc=True):
        super().__init__()
        self.base_model = base_model
        self.use_gc = use_gc

    def forward(self, input_ids, labels=None, **kwargs):
        hidden_states = self.base_model.embedding(input_ids)
        
        for i, block in enumerate(self.base_model.blocks):
            shift_size = (block.C // 2) if (i % 2 != 0) else 0
            
            if self.training and self.use_gc:
                hidden_states = checkpoint.checkpoint(
                    block, hidden_states, shift_size, use_reentrant=False
                )
            else:
                hidden_states = block(hidden_states, shift_size=shift_size)
                
        hidden_states = self.base_model.final_norm(hidden_states)
        
        loss = None
        logits = None
        
        if labels is not None:
            # 🚀 SOTA 記憶體突破：Chunked Cross-Entropy
            # 不再一次性計算 15.6 GiB 的 full logits 矩陣
            shift_hidden_states = hidden_states[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss_fct = nn.CrossEntropyLoss(reduction="sum")
            total_loss = 0.0
            
            # 將 32K 切割為 2048 的小塊來分批計算 Loss
            chunk_size = 2048 
            seq_len = shift_hidden_states.size(1)
            
            for i in range(0, seq_len, chunk_size):
                end_idx = min(i + chunk_size, seq_len)
                
                # 只生成一小塊的 Logits，VRAM 消耗極低
                chunk_hidden = shift_hidden_states[:, i:end_idx, :]
                chunk_logits = self.base_model.fc_out(chunk_hidden)
                chunk_labels = shift_labels[:, i:end_idx]
                
                # 強制使用 float32 確保 Loss 計算不溢位
                chunk_loss = loss_fct(
                    chunk_logits.view(-1, chunk_logits.size(-1)).float(), 
                    chunk_labels.view(-1)
                )
                total_loss += chunk_loss
                
                # 手動釋放小塊的計算圖
                del chunk_logits, chunk_hidden
                
            # 平均 Loss
            loss = total_loss / shift_labels.numel()
        else:
            # 推論時 (無 labels) 才需要完整輸出 logits
            logits = self.base_model.fc_out(hidden_states)
            
        # 訓練模式下不回傳 logits 給 HF Trainer，避免它又雞婆去 alloc 15.6 GiB
        if logits is None:
            return {"loss": loss}
            
        return {"loss": loss, "logits": logits}
# ==========================================
# 2. 權重精準映射與凍結邏輯
# ==========================================

def transplant_and_freeze(base_model_id, agiv2_base, num_layers=32):
    print(f"🔄 開始從 {base_model_id} 萃取知識庫權重...")
    
    # 針對沒有審核限制的模型，直接下載並以 CPU 載入
    source_model = AutoModelForCausalLM.from_pretrained(
        base_model_id, torch_dtype=torch.bfloat16, device_map="cpu"
    )
    source_sd = source_model.state_dict()
    new_sd = {}

    # 1. 首尾映射
    if 'model.embed_tokens.weight' in source_sd:
        new_sd['embedding.weight'] = source_sd['model.embed_tokens.weight']
    if 'lm_head.weight' in source_sd:
        new_sd['fc_out.weight'] = source_sd['lm_head.weight']
    if 'model.norm.weight' in source_sd:
        new_sd['final_norm.weight'] = source_sd['model.norm.weight']

    # 2. 逐層映射 (支援 1:3 稀疏架構)
    for i in range(num_layers):
        src_prefix = f"model.layers.{i}."
        agiv2_prefix = f"blocks.{i}."
        
        is_global = (i + 1) % 4 == 0
        
        # 處理 RMSNorm 語意對齊
        if is_global:
            new_sd[f"{agiv2_prefix}norm2.weight"] = source_sd.get(f"{src_prefix}input_layernorm.weight")
            new_sd[f"{agiv2_prefix}norm3.weight"] = source_sd.get(f"{src_prefix}post_attention_layernorm.weight")
        else:
            new_sd[f"{agiv2_prefix}norm1.weight"] = source_sd.get(f"{src_prefix}input_layernorm.weight")
            new_sd[f"{agiv2_prefix}norm2.weight"] = source_sd.get(f"{src_prefix}post_attention_layernorm.weight")
            
        # 處理 SwiGLU FFN
        new_sd[f"{agiv2_prefix}ffn.gate_proj.weight"] = source_sd.get(f"{src_prefix}mlp.gate_proj.weight")
        new_sd[f"{agiv2_prefix}ffn.up_proj.weight"] = source_sd.get(f"{src_prefix}mlp.up_proj.weight")
        new_sd[f"{agiv2_prefix}ffn.down_proj.weight"] = source_sd.get(f"{src_prefix}mlp.down_proj.weight")

    # 3. 匯入權重並將未匹配的新器官轉為 bfloat16
    new_sd = {k: v for k, v in new_sd.items() if v is not None}
    missing_keys, _ = agiv2_base.load_state_dict(new_sd, strict=False)
    
    # 4. 執行物理凍結
    frozen_params = 0
    trainable_params = 0
    for name, param in agiv2_base.named_parameters():
        if name in new_sd:
            param.requires_grad = False
            frozen_params += param.numel()
        else:
            param.requires_grad = True
            param.data = param.data.to(torch.bfloat16)
            trainable_params += param.numel()

    print(f"❄️ 成功凍結知識庫參數: {frozen_params / 1e9:.2f} B")
    print(f"🔥 AGIV2L 新生路由器官參數 (Trainable): {trainable_params / 1e9:.2f} B")
    
    del source_model
    import gc
    gc.collect()
    
    return agiv2_base

# ==========================================
# 3. 32K SOTA 串流與 Packing 資料集
# ==========================================

# ==========================================
# 3. 32K SOTA 串流與 Packing 資料集 (原生 Parquet 升級版)
# ==========================================

class Packed32KDataset(torch.utils.data.IterableDataset):
    def __init__(self, tokenizer, max_length=32768):
        # 1. FineWeb-Edu: 高品質教育文本
        ds_edu = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
        
        # 2. Cosmopedia: 替換為 'stanford' (史丹佛課程資料) 或 'openstax' (大學教科書)
        ds_cosmo = load_dataset("HuggingFaceTB/cosmopedia", name="stanford", split="train", streaming=True)
        
        # 混合這兩個絕對安全的資料源
        self.dataset = interleave_datasets([ds_edu, ds_cosmo], probabilities=[0.6, 0.4])
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __iter__(self):
        buffer = []
        for example in self.dataset:
            # 確保提取文字欄位
            text_content = example.get('text', '')
            if not text_content:
                continue
                
            tokens = self.tokenizer(text_content, add_special_tokens=False)['input_ids']
            tokens.append(self.tokenizer.eos_token_id)
            buffer.extend(tokens)
            
            while len(buffer) >= self.max_length:
                chunk = buffer[:self.max_length]
                buffer = buffer[self.max_length:]
                yield {
                    "input_ids": torch.tensor(chunk, dtype=torch.long),
                    "labels": torch.tensor(chunk, dtype=torch.long)
                }

# ==========================================
# 4. 啟動煉丹爐
# ==========================================

def main():
    # 🚀 更換為無審核限制的 LLaMA-3 8B 版本
    base_model_id = "NousResearch/Meta-Llama-3-8B" 
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("🏗️ 建立 AGIV2L 實體 (1:3 稀疏架構, D=4096, M=1024, C=1024)...")
    agiv2_base = AGIV2L(
        vocab_size=len(tokenizer), 
        D=4096, 
        hidden_dim=14336, 
        num_blocks=32,
        C=1024, K=1024, M=1024
    )
    
    agiv2_base = transplant_and_freeze(base_model_id, agiv2_base, num_layers=32)
    
    print("🔌 啟動 Gradient Checkpointing 並傳送至 CUDA:1 環境...")
    model = AGIV2LForCausalLM(agiv2_base, use_gc=True).to(torch.bfloat16).cuda()

    print("📚 連結 RedPajama 32K 論文與書籍資料串流...")
    train_dataset = Packed32KDataset(tokenizer, max_length=32768)

    training_args = TrainingArguments(
        output_dir="./agiv2-llama3-cpt-32k",
        max_steps=5000,                
        per_device_train_batch_size=1, 
        gradient_accumulation_steps=8, 
        
        learning_rate=1e-4,            
        warmup_ratio=0.05,             
        lr_scheduler_type="cosine",    
        
        weight_decay=0.01,
        bf16=True,
        logging_steps=10,
        save_steps=500,
        
        optim="paged_adamw_8bit",
        report_to="none",
        gradient_checkpointing=False, 
        dataloader_num_workers=2       
    )

    print("🚀 引擎點火：啟動 32K 長文本 Continual Pre-Training...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )

    trainer.train()
    
    torch.save(model.state_dict(), "./agiv2-llama3-cpt-32k/final_cpt_state.pth")
    print("🎉 32K CPT 訓練圓滿結束！AGIV2L 已具備超長文本的全局記憶檢索能力。")

if __name__ == "__main__":
    main()