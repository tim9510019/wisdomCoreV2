"""
train32KAC.py — 四通道混合 32K CPT 物理隔離版 (N -> N+B 範式)
===========================================================
核心機制：
1. 揚棄 Teacher Forcing：不再計算全序列 32K 的 Next-Token Loss。
2. Chunk-wise 物理隔離：將每筆長文本切分為 N (上文) 與 B (目標 Chunk)。
3. 強制遮蔽：輸入端將 B 的位置全部替換為 <PAD>，杜絕雙向架構的未來洩漏。
4. 拓撲引導梯度：僅對 B 區塊 (長度 1024) 計算 Loss，以高質量的全局梯度取代盲目猜測。
"""

import os
import sys
import csv
import time
import torch

# 確保基礎元件路徑正確
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# ==========================================
# 0. SOTA 環境配置
# ==========================================
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["BITSANDBYTES_NOWELCOME"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from transformers import AutoTokenizer, TrainingArguments, Trainer, set_seed, TrainerCallback
from transformers.trainer_utils import get_last_checkpoint
from datasets import load_from_disk 

# 🌟 確保載入的是無因果遮罩的 AC 版本
from AGIV2GAC import AGIV2G
from trainAGI import AGIV2GForCausalLM, transplant_and_freeze

set_seed(42)

# ==========================================
# 1. 具備歷史回溯能力的動態監控
# ==========================================
class AGIV2GMonitor(TrainerCallback):
    def __init__(self, path="./agiv2_cpt_32k_ac_log.csv", save_dir="./agiv2-cpt-32k-ac"):
        self.path = path
        self.save_dir = save_dir
        self.best_eval_loss = float('inf')
        
        os.makedirs(self.save_dir, exist_ok=True)
        
        if os.path.exists(self.path):
            try:
                with open(self.path, 'r', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    header = next(reader, None) 
                    if header:
                        for row in reader:
                            if len(row) > 2 and row[2].strip():
                                try:
                                    val = float(row[2])
                                    if val < self.best_eval_loss:
                                        self.best_eval_loss = val
                                except ValueError: pass
                if self.best_eval_loss != float('inf'):
                    print(f"\n📈 [Monitor] 成功讀取歷史紀錄！當前最佳 Eval Loss 基準線為: {self.best_eval_loss:.4f}")
            except Exception as e:
                print(f"\n⚠️ [Monitor] 讀取紀錄失敗: {e}，將重置基準線。")
        else:
            with open(self.path, 'w', newline='') as f:
                csv.writer(f).writerow(['step', 'train_loss', 'eval_loss', 'lr', 'gate_fft', 'gate_mem', 'time'])

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            model = kwargs.get('model', None)
            avg_gate_fft, avg_gate_mem = 0.0, 0.0
            
            if model is not None:
                core_model = model.module.base_model if hasattr(model, 'module') else model.base_model
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
                    state.global_step, logs.get("loss", ""), logs.get("eval_loss", ""), 
                    logs.get("learning_rate", ""), 
                    f"{avg_gate_fft:.6f}" if gate_fft_vals else "",
                    f"{avg_gate_mem:.6f}" if gate_fft_vals else "", time.ctime()
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
                    raw_model = model.module if hasattr(model, 'module') else model
                    torch.save(raw_model.state_dict(), best_model_path)
                    print(f"\n[Monitor] 🌟 Eval Loss 進步 ({old_best:.4f} -> {current_eval_loss:.4f})，已淬鍊權重至 {best_model_path}")
            else:
                print(f"\n[Monitor] 🛡️ 此次 Eval Loss ({current_eval_loss:.4f}) 未超越最佳 ({self.best_eval_loss:.4f})。")

# ==========================================
# 2. 自定義 CPT 訓練器：N -> N+B 物理隔離
# ==========================================
class CPTTrainerAC(Trainer):
    def __init__(self, pad_token_id=0, b_size=1024, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pad_token_id = pad_token_id
        self.b_size = b_size # 目標預測 Chunk 的長度 B

    def _physical_isolation_forward(self, model, input_ids):
        """執行大範圍文本的 N -> N+B 隔離與計算"""
        L = input_ids.shape[1]
        
        # 保證至少留有 3/4 的長度作為 N 進行全局理解，B 最高不超過 self.b_size
        b_actual = min(self.b_size, L // 4) if L > self.b_size else L // 4
        N_len = L - b_actual
        
        # [輸入隔離] 把最後 B 個字元覆蓋為 PAD，徹底斷絕未來視野
        inputs_isolated = input_ids.clone()
        inputs_isolated[:, N_len:] = self.pad_token_id
        
        # [梯度對焦] 將 N 的標籤設為 -100，僅要求模型精準預測最後的 B
        labels = torch.full_like(input_ids, -100)
        labels[:, N_len:] = input_ids[:, N_len:]
        
        return model(input_ids=inputs_isolated, labels=labels)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        input_ids = inputs.get("input_ids")
        # 由於是 CPT，這裡不再分 pos/neg，每一筆資料都是龐大的異質文本
        outputs = self._physical_isolation_forward(model, input_ids)
        loss = outputs["loss"]
        
        # 可選：在 CPT 階段維持輕微的門控激活壓力，避免退化
        raw_model = model.module if hasattr(model, 'module') else model
        core_model = raw_model.base_model if hasattr(raw_model, 'base_model') else raw_model
        gate_penalty = 0.0
        for b in core_model.blocks:
            if hasattr(b, 'gate_mem'):
                gate_penalty += torch.exp(-torch.abs(b.gate_mem) * 10.0).mean()
                
        total_loss = loss + 0.05 * gate_penalty
        return (total_loss, outputs) if return_outputs else total_loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            loss = self.compute_loss(model, inputs, return_outputs=False)
        if prediction_loss_only: 
            return (loss, None, None)
            
        input_ids = inputs.get("input_ids")
        L = input_ids.shape[1]
        b_actual = min(self.b_size, L // 4) if L > self.b_size else L // 4
        N_len = L - b_actual
        
        # 評估階段同樣執行物理隔離
        inputs_iso = input_ids.clone()
        inputs_iso[:, N_len:] = self.pad_token_id
        
        with torch.no_grad():
            outputs = model(input_ids=inputs_iso, labels=input_ids)
            
        return (loss, outputs.logits, input_ids)

# ==========================================
# 3. 主流程
# ==========================================
def main():
    model_id = "google/gemma-3-1b-it"
    save_dir = "./agiv2-cpt-32k-ac" # 獨立儲存路徑，避免與舊版混淆
    data_dir = "./agiv2_pretokenized_32k_mixed" 
    
    os.makedirs(save_dir, exist_ok=True)

    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"找不到預處理資料集路徑 {data_dir}。請先執行 prepare_data_32k.py")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    pad_id = tokenizer.pad_token_id

    # 初始化 AGIV2GAC 無遮罩實體架構
    base = AGIV2G(vocab_size=262144, D=1152, hidden_dim=6912, num_blocks=26)
    base = transplant_and_freeze(model_id, base)
    model = AGIV2GForCausalLM(base).to(torch.bfloat16).cuda()

    # Roadmap 階段二解凍策略：Unfreeze Routing + LayerNorm
    unlock_keys = ["gate_fft", "gate_mem", "omegas", "mlp_H", "fft_norm", "Q_mem", "W_k_mem", "W_v_mem", "mem_norm", "W_q_cross", "o_proj_cross", "layernorm"]
    for name, param in model.named_parameters():
        param.requires_grad = any(k in name for k in unlock_keys)

    print(f"\n📦 讀取 Memory-Mapped 預處理四通道資料集 ({data_dir})...")
    processed_datasets = load_from_disk(data_dir)
    train_ds = processed_datasets["train"]
    val_ds = processed_datasets["test"]

    # 對齊 Roadmap 階段二參數 (LR=1e-4, Steps=5000)
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
        save_steps=10, 
        save_total_limit=2, 
        eval_strategy="steps",
        eval_steps=100,
        optim="paged_adamw_8bit",
        report_to="none",
        gradient_checkpointing=False,
        dataloader_num_workers=4, 
    )

    trainer = CPTTrainerAC(
        pad_token_id=pad_id,
        b_size=1024, # 🌟 指定 Chunk 長度為 1024 進行預測
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        callbacks=[AGIV2GMonitor(save_dir=save_dir)] 
    )

    last_checkpoint = get_last_checkpoint(save_dir)
    if last_checkpoint is not None:
        print(f"\n🚀 偵測到中斷點 {last_checkpoint}，啟動 AC 狀態無損恢復...")
    else:
        print("\n🚀 開始 32K 物理隔離版 (N -> N+B) CPT 全新訓練 (CUDA:0)...")

    trainer.train(resume_from_checkpoint=last_checkpoint)

    torch.save(model.state_dict(), os.path.join(save_dir, "final_ac.pth"))
    print("🎉 物理隔離 CPT 訓練完成！")

if __name__ == "__main__":
    main()