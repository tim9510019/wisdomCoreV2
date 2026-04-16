"""
train32K.py — 四通道混合 32K CPT 訓練腳本 (預先處理極速版)
===============================================
"""

import os
import sys
import csv
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ==========================================
# 0. SOTA 環境配置 (必須在 import torch 之前)
# ==========================================
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["BITSANDBYTES_NOWELCOME"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from transformers import AutoTokenizer, TrainingArguments, Trainer, set_seed, TrainerCallback
from transformers.trainer_utils import get_last_checkpoint
from datasets import load_from_disk 

from AGIV2G import AGIV2G
# 移除引入舊版的 AGIV2GMonitor，其餘維持不變
from train import AGIV2GForCausalLM, transplant_and_freeze

set_seed(42)

# ==========================================
# 1. 具備歷史回溯能力的動態監控 (解決亂存檔盲點)
# ==========================================
class AGIV2GMonitor(TrainerCallback):
    def __init__(self, path="./agiv2_cpt_32k_log.csv", save_dir="./agiv2-cpt-32k"):
        self.path = path
        self.save_dir = save_dir
        self.best_eval_loss = float('inf')
        
        os.makedirs(self.save_dir, exist_ok=True)
        
        # 🌟 啟動時掃描歷史 CSV，找回最佳 Eval Loss 基準
        if os.path.exists(self.path):
            try:
                with open(self.path, 'r', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    header = next(reader, None) # 跳過標題列
                    if header:
                        for row in reader:
                            # row[2] 是 eval_loss
                            if len(row) > 2 and row[2].strip():
                                try:
                                    val = float(row[2])
                                    if val < self.best_eval_loss:
                                        self.best_eval_loss = val
                                except ValueError:
                                    pass
                
                if self.best_eval_loss != float('inf'):
                    print(f"\n📈 [Monitor] 成功讀取歷史紀錄！當前最佳 Eval Loss 基準線為: {self.best_eval_loss:.4f}")
            except Exception as e:
                print(f"\n⚠️ [Monitor] 讀取歷史紀錄失敗: {e}，將重置基準線。")
        else:
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
            else:
                # 拒絕亂覆蓋，印出提示
                print(f"\n[Monitor] 🛡️ 此次 Eval Loss ({current_eval_loss:.4f}) 未超越歷史最佳 ({self.best_eval_loss:.4f})，跳過權重保存。")

# ==========================================
# 2. 主流程
# ==========================================

def main():
    model_id = "google/gemma-3-1b-it"
    save_dir = "./agiv2-cpt-32k"
    data_dir = "./agiv2_pretokenized_32k_mixed" # 完美對接階段一產出的四通道資料集
    
    os.makedirs(save_dir, exist_ok=True)

    # 防呆機制：確保資料真的存在才啟動模型載入
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"找不到預處理資料集路徑 {data_dir}。請先執行 prepare_data_32k.py")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    base = AGIV2G(vocab_size=262144, D=1152, hidden_dim=6912, num_blocks=26)
    base = transplant_and_freeze(model_id, base)

    model = AGIV2GForCausalLM(base).to(torch.bfloat16).cuda()

    print(f"\n📦 讀取 Memory-Mapped 預處理四通道資料集 ({data_dir})...")
    # 這裡的載入延遲近乎為 0，因為依賴 Arrow 的底層映射
    processed_datasets = load_from_disk(data_dir)
    train_ds = processed_datasets["train"]
    val_ds = processed_datasets["test"]

    args = TrainingArguments(
        output_dir=save_dir,
        max_steps=40000,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=1e-4,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        bf16=True,
        logging_steps=1,
        save_steps=100, # 🌟 關鍵效能修正：避免每步存檔拖垮 I/O，對齊 eval_steps
        save_total_limit=2, # 僅保留最後 2 個 Checkpoint，節省硬碟空間
        eval_strategy="steps",
        eval_steps=100,
        optim="paged_adamw_8bit",
        report_to="none",
        gradient_checkpointing=False,
        dataloader_num_workers=4, # 🌟 核心突破：利用 4 個 CPU Worker 瘋狂餵 32K 資料給 GPU
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        callbacks=[AGIV2GMonitor(save_dir=save_dir)] # 改用本地具備記憶機制的 Monitor
    )

    last_checkpoint = get_last_checkpoint(save_dir)
    if last_checkpoint is not None:
        print(f"\n🚀 偵測到極限中斷點 {last_checkpoint}，啟動全維度狀態無損恢復...")
    else:
        print("\n🚀 開始 32K 四通道混合 CPT 全新訓練 (CUDA:0)...")

    trainer.train(resume_from_checkpoint=last_checkpoint)

    torch.save(model.state_dict(), os.path.join(save_dir, "final.pth"))
    print("🎉 訓練完成！")


if __name__ == "__main__":
    main()