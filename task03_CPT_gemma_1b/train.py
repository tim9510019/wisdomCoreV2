import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ==========================================
# 0. SOTA 環境配置 (必須在 import torch 之前)
# ==========================================
# 完全對齊 train.py: 使用設備 "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
os.environ["BITSANDBYTES_NOWELCOME"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, TrainerCallback, set_seed
from transformers.trainer_utils import get_last_checkpoint
from datasets import load_dataset, interleave_datasets
import math
import csv
import time
import shutil # 對齊 train.py: 補回 shutil

from AGIV2G import AGIV2G

set_seed(42)

# ==========================================
# 1. 訓練包裝器 (動態 GC & Chunked Loss)
# ==========================================

class AGIV2GForCausalLM(nn.Module):
    def __init__(self, base_model, use_gc=True):
        super().__init__()
        self.base_model = base_model
        self.use_gc = use_gc

    def forward(self, input_ids, labels=None, **kwargs):
        hidden_states = self.base_model.embedding(input_ids)
        # 保留 Gemma 架構必須的縮放
        hidden_states = hidden_states * math.sqrt(self.base_model.D)
        
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
            shift_hidden = hidden_states[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss_fct = nn.CrossEntropyLoss(reduction="sum", ignore_index=-100)
            total_loss = 0.0
            valid_tokens = 0.0 
            chunk_size = 2048 
            seq_len = shift_hidden.size(1)
            
            for i in range(0, seq_len, chunk_size):
                end_idx = min(i + chunk_size, seq_len)
                c_hidden = shift_hidden[:, i:end_idx, :]
                c_logits = self.base_model.fc_out(c_hidden)
                c_labels = shift_labels[:, i:end_idx]
                
                c_loss = loss_fct(c_logits.reshape(-1, c_logits.size(-1)).float(), c_labels.reshape(-1))
                total_loss += c_loss
                
                valid_tokens += (c_labels != -100).sum().item()
                del c_logits, c_hidden
                
            loss = total_loss / max(valid_tokens, 1)
        else:
            logits = self.base_model.fc_out(hidden_states)
            
        return {"loss": loss, "logits": logits} if logits is not None else {"loss": loss}

# ==========================================
# 2. 權重移植、凍結與動態監控 Callback
# ==========================================

def transplant_and_freeze(model_id, agiv2_base):
    print(f"🔄 從 {model_id} 移植知識庫 (使用 dtype=bf16)...")
    src_model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map="cpu"
    )
    src_sd = src_model.state_dict()
    new_sd = {}

    new_sd['embedding.weight'] = src_sd.get('model.embed_tokens.weight')
    # 保留 Gemma 可能的權重綁定邏輯
    new_sd['fc_out.weight'] = src_sd.get('lm_head.weight') if 'lm_head.weight' in src_sd else src_sd.get('model.embed_tokens.weight')
    new_sd['final_norm.weight'] = src_sd.get('model.norm.weight')

    for i in range(26):
        s_pre = f"model.layers.{i}."
        a_pre = f"blocks.{i}."
        
        new_sd[f"{a_pre}input_layernorm.weight"] = src_sd.get(f"{s_pre}input_layernorm.weight")
        new_sd[f"{a_pre}post_attention_layernorm.weight"] = src_sd.get(f"{s_pre}post_attention_layernorm.weight")
        new_sd[f"{a_pre}pre_feedforward_layernorm.weight"] = src_sd.get(f"{s_pre}pre_feedforward_layernorm.weight")
        new_sd[f"{a_pre}post_feedforward_layernorm.weight"] = src_sd.get(f"{s_pre}post_feedforward_layernorm.weight")
        
        new_sd[f"{a_pre}q_norm.weight"] = src_sd.get(f"{s_pre}self_attn.q_norm.weight")
        new_sd[f"{a_pre}k_norm.weight"] = src_sd.get(f"{s_pre}self_attn.k_norm.weight")
        
        new_sd[f"{a_pre}ffn.gate_proj.weight"] = src_sd.get(f"{s_pre}mlp.gate_proj.weight")
        new_sd[f"{a_pre}ffn.up_proj.weight"] = src_sd.get(f"{s_pre}mlp.up_proj.weight")
        new_sd[f"{a_pre}ffn.down_proj.weight"] = src_sd.get(f"{s_pre}mlp.down_proj.weight")

        new_sd[f"{a_pre}W_q_loc.weight"] = src_sd.get(f"{s_pre}self_attn.q_proj.weight")
        new_sd[f"{a_pre}W_k_loc.weight"] = src_sd.get(f"{s_pre}self_attn.k_proj.weight")
        new_sd[f"{a_pre}W_v_loc.weight"] = src_sd.get(f"{s_pre}self_attn.v_proj.weight")
        new_sd[f"{a_pre}o_proj_loc.weight"] = src_sd.get(f"{s_pre}self_attn.o_proj.weight")

    new_sd = {k: v for k, v in new_sd.items() if v is not None}
    agiv2_base.load_state_dict(new_sd, strict=False)
    
    frozen_count, trainable_count = 0, 0
    for name, param in agiv2_base.named_parameters():
        param.requires_grad = name not in new_sd
        if param.requires_grad:
            param.data = param.data.to(torch.bfloat16)
            trainable_count += param.numel()
        else:
            frozen_count += param.numel()

    print(f"✅ 知識庫移植完成！")
    print(f"🔒 凍結 Gemma 3 參數: {frozen_count:,} | 🟢 新架構可訓練參數: {trainable_count:,}")

    del src_model
    import gc; gc.collect(); torch.cuda.empty_cache()
    return agiv2_base

class AGIV2GMonitor(TrainerCallback):
    def __init__(self, path="./agiv2_cpt_log.csv", save_dir="./agiv2-cpt"):
        self.path = path
        self.save_dir = save_dir
        self.best_eval_loss = float('inf')
        
        os.makedirs(self.save_dir, exist_ok=True)
        
        if not os.path.exists(self.path):
            with open(self.path, 'w', newline='') as f:
                csv.writer(f).writerow(['step', 'train_loss', 'eval_loss', 'lr', 'gate_fft', 'gate_mem', 'time'])

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            model = kwargs.get('model', None)
            avg_gate_fft, avg_gate_mem = 0.0, 0.0
            
            if model is not None:
                core_model = model.base_model if hasattr(model, 'base_model') else model
                gate_fft_vals, gate_mem_vals = [], []
                
                for block in core_model.blocks:
                    if hasattr(block, 'gate_fft'):
                        gate_fft_vals.append(block.gate_fft.item())
                        gate_mem_vals.append(block.gate_mem.item())
                
                if gate_fft_vals:
                    avg_gate_fft = sum(gate_fft_vals) / len(gate_fft_vals)
                    avg_gate_mem = sum(gate_mem_vals) / len(gate_mem_vals)
                    logs["gate_fft"] = round(avg_gate_fft, 6)
                    logs["gate_mem"] = round(avg_gate_mem, 6)

            with open(self.path, 'a', newline='') as f:
                csv.writer(f).writerow([
                    state.global_step, 
                    logs.get("loss", ""), 
                    logs.get("eval_loss", ""), 
                    logs.get("learning_rate", ""), 
                    f"{avg_gate_fft:.6f}" if gate_fft_vals else "",
                    f"{avg_gate_mem:.6f}" if gate_fft_vals else "",
                    time.ctime()
                ])

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics and "eval_loss" in metrics:
            current_eval_loss = metrics["eval_loss"]
            if current_eval_loss < self.best_eval_loss:
                old_best = self.best_eval_loss
                self.best_eval_loss = current_eval_loss
                model = kwargs.get('model', None)
                if model is not None:
                    best_model_path = os.path.join(self.save_dir, "best_model.pth")
                    torch.save(model.state_dict(), best_model_path)
                    print(f"\n[Monitor] 🌟 Eval Loss 進步 ({old_best:.4f} -> {current_eval_loss:.4f})，已淬鍊權重至 {best_model_path}")

# ==========================================
# 3. 資料集隔離 (Train: 前段, Val: 後段)
# ==========================================

class Packed32KDataset(torch.utils.data.IterableDataset):
    def __init__(self, tokenizer, max_length=32768, is_val=False, max_samples=None):
        ds_edu = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=False)
        ds_cos = load_dataset("HuggingFaceTB/cosmopedia", name="stanford", split="train", streaming=False)
        
        if is_val:
            # 驗證集：只拿前面的 1000 筆跟 500 筆資料就足夠評估了
            ds_edu, ds_cos = ds_edu.take(1000), ds_cos.take(500)
        else:
            # 訓練集：跳過前面給驗證集用掉的資料後，剩餘的「無限供應」，不設 .take() 上限！
            ds_edu, ds_cos = ds_edu.skip(1000), ds_cos.skip(500)

        self.dataset = interleave_datasets([ds_edu, ds_cos], probabilities=[0.6, 0.4])
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.max_samples = max_samples 

    def __iter__(self):
        buffer = []
        yield_count = 0 
        
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
                yield_count += 1
                
                if self.max_samples is not None and yield_count >= self.max_samples:
                    return

# ==========================================
# 4. 主流程
# ==========================================

def main():
    model_id = "google/gemma-3-1b-it"
    save_dir = "./agiv2-cpt"
    os.makedirs(save_dir, exist_ok=True)
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    base = AGIV2G(vocab_size=262144, D=1152, hidden_dim=6912, num_blocks=26)
    base = transplant_and_freeze(model_id, base)
    
    # 完全對齊 train.py：隱式使用預設的 use_gc=True，移除傳遞參數
    model = AGIV2GForCausalLM(base).to(torch.bfloat16).cuda()

    train_ds = Packed32KDataset(tokenizer, max_length=32768, is_val=False)
    val_ds = Packed32KDataset(tokenizer, max_length=32768, is_val=True, max_samples=20) 

    args = TrainingArguments(
        output_dir=save_dir,
        max_steps=5000, 
        per_device_train_batch_size=1, 
        gradient_accumulation_steps=8,
        learning_rate=1e-4, 
        warmup_ratio=0.05, 
        lr_scheduler_type="cosine",
        weight_decay=0.01, 
        bf16=True, 
        logging_steps=1, 
        save_steps=1, 
        save_total_limit=2, 
        eval_strategy="steps", 
        eval_steps=100,
        optim="paged_adamw_8bit", 
        report_to="none", 
        gradient_checkpointing=False
    )

    trainer = Trainer(
        model=model, 
        args=args, 
        train_dataset=train_ds, 
        eval_dataset=val_ds, 
        callbacks=[AGIV2GMonitor(save_dir=save_dir)]
    )
    
    last_checkpoint = get_last_checkpoint(save_dir)
    if last_checkpoint is not None:
        print(f"\n🚀 偵測到極限中斷點 {last_checkpoint}，啟動全維度狀態無損恢復...")
    else:
        print("\n🚀 開始 32K CPT 全新訓練 (CUDA:0)...")

    trainer.train(resume_from_checkpoint=last_checkpoint)
    
    torch.save(model.state_dict(), os.path.join(save_dir, "final.pth"))
    print("🎉 訓練完成！")

if __name__ == "__main__":
    main()