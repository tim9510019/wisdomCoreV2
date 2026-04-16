"""
trainRouter_V3_Final_Fixed.py — 路由激發終極腳本 (同步 train32K 儲存機制)
===========================================================
修正項目：
1. 具備歷史回溯能力的 Monitor：重啟時自動讀取 CSV 找回最佳 Eval Loss。
2. 全自動斷點續傳：自動偵測 checkpoint 並恢復訓練狀態。
3. 儲存機制對齊：save_steps 與 save_total_limit 最佳化，防止硬碟爆滿。
"""
import os
import sys
import csv
import time
import torch
from transformers import AutoTokenizer, TrainingArguments, Trainer, set_seed, TrainerCallback
from transformers.trainer_utils import get_last_checkpoint
from datasets import load_from_disk, Dataset

# 確保基礎元件路徑正確
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from AGIV2G import AGIV2G
from trainAGI import AGIV2GForCausalLM, transplant_and_freeze

set_seed(2026)

# ==========================================
# 1. 具備記憶能力的監控器 (遵循 train32K.py)
# ==========================================
class ZeroGateMonitor(TrainerCallback):
    def __init__(self, path="./zerogate_32k_log.csv", save_dir="./agiv2_zerogate_checkpoints"):
        self.path = path
        self.save_dir = save_dir
        self.best_eval_loss = float('inf')
        os.makedirs(self.save_dir, exist_ok=True)
        
        # 🌟 歷史回溯：掃描 CSV 找回之前的最佳成績
        if os.path.exists(self.path):
            try:
                with open(self.path, 'r', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    next(reader, None) # 跳過標題
                    for row in reader:
                        if len(row) > 2 and row[2].strip(): # row[2] 是 eval_loss
                            try:
                                val = float(row[2])
                                if val < self.best_eval_loss:
                                    self.best_eval_loss = val
                            except ValueError: pass
                if self.best_eval_loss != float('inf'):
                    print(f"\n📈 [Monitor] 成功找回歷史紀錄！當前最佳 Eval Loss 基準: {self.best_eval_loss:.4f}")
            except Exception as e:
                print(f"\n⚠️ [Monitor] 讀取歷史紀錄失敗: {e}")
        else:
            with open(self.path, 'w', newline='') as f:
                csv.writer(f).writerow(['step', 'loss', 'eval_loss', 'fft_max', 'mem_max', 'time'])

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            model = kwargs.get('model', None)
            max_fft, max_mem = 0.0, 0.0
            if model is not None:
                raw_model = model.module if hasattr(model, 'module') else model
                core_model = raw_model.base_model if hasattr(raw_model, 'base_model') else raw_model
                g_fft = [b.gate_fft.item() for b in core_model.blocks if hasattr(b, 'gate_fft')]
                g_mem = [b.gate_mem.item() for b in core_model.blocks if hasattr(b, 'gate_mem')]
                if g_fft:
                    max_fft, max_mem = max(g_fft), max(g_mem)
                    logs["fft_max"] = round(max_fft, 6)
                    logs["mem_max"] = round(max_mem, 6)
            
            with open(self.path, 'a', newline='') as f:
                csv.writer(f).writerow([state.global_step, logs.get("loss", ""), logs.get("eval_loss", ""), max_fft, max_mem, time.ctime()])

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """🌟 嚴格保存最佳模型 (拒絕亂覆蓋)"""
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
                    print(f"\n[Monitor] 🌟 發現更佳權重 ({old_best:.4f} -> {current_eval_loss:.4f})，已儲存至 {best_model_path}")
            else:
                print(f"\n[Monitor] 🛡️ 此次成績 ({current_eval_loss:.4f}) 未超越歷史最佳 ({self.best_eval_loss:.4f})，跳過保存。")

# ==========================================
# 2. 自定義 Trainer：解決維度與評估衝突
# ==========================================
class RouterTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        pos_ids, neg_ids = inputs.get("pos_ids"), inputs.get("neg_ids")
        with torch.no_grad():
            neg_loss = model(input_ids=neg_ids, labels=neg_ids)["loss"].detach()
        pos_outputs = model(input_ids=pos_ids, labels=pos_ids)
        pos_loss = pos_outputs["loss"]
        
        raw_model = model.module if hasattr(model, 'module') else model
        core_model = raw_model.base_model if hasattr(raw_model, 'base_model') else raw_model
        
        gate_penalty = 0.0
        for b in core_model.blocks:
            if hasattr(b, 'gate_mem'):
                gate_penalty = gate_penalty + torch.exp(-torch.abs(b.gate_mem) * 20.0).mean()
        
        contrastive_loss = torch.clamp(pos_loss - neg_loss + 1.0, min=0.0).mean()
        total_loss = pos_loss + 0.5 * contrastive_loss + 0.1 * gate_penalty
        return (total_loss.mean(), pos_outputs) if return_outputs else total_loss.mean()

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            loss = self.compute_loss(model, inputs, return_outputs=False)
        if prediction_loss_only: return (loss, None, None)
        with torch.no_grad():
            outputs = model(input_ids=inputs.get("pos_ids"), labels=inputs.get("pos_ids"))
        return (loss, outputs.logits, inputs.get("pos_ids"))

# ==========================================
# 3. 主流程
# ==========================================
def main():
    model_id = "google/gemma-3-1b-it"
    save_dir = "./agiv2_zerogate_checkpoints"
    data_dir = "./agiv2_router_v3_max"
    
    raw_ds = load_from_disk(data_dir)
    if isinstance(raw_ds, Dataset):
        ds = raw_ds.train_test_split(test_size=0.05, seed=2026)
    else:
        ds = raw_ds

    # EVAL 50 筆限制
    eval_limit = min(50, len(ds["test"]))
    ds["test"] = ds["test"].select(range(eval_limit))

    # 初始化架構
    base = AGIV2G(vocab_size=262144, D=1152, hidden_dim=6912, num_blocks=26)
    base = transplant_and_freeze(model_id, base)
    model = AGIV2GForCausalLM(base, use_gc=True)
    
    unlock_keywords = ["gate_fft", "gate_mem", "omegas", "mlp_H", "fft_norm", "Q_mem", "W_k_mem", "W_v_mem", "mem_norm", "W_q_cross", "o_proj_cross"]
    for name, param in model.named_parameters():
        param.requires_grad = any(k in name for k in unlock_keywords)
    
    for name, param in model.named_parameters():
        if "gate_fft" in name or "gate_mem" in name:
            param.register_hook(lambda grad: grad * 10.0)
            
    model = model.cuda().to(torch.bfloat16)

    # 🌟 設定與 train32K.py 對齊
    args = TrainingArguments(
        output_dir=save_dir,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=32,
        learning_rate=3e-4,
        lr_scheduler_type="cosine",
        bf16=True,
        logging_steps=1,
        eval_strategy="steps",
        eval_steps=50,
        save_steps=10,                # 🌟 每 10 步存一次 Checkpoint (對齊評測頻率)
        save_total_limit=2,           # 🌟 僅保留 2 個，節省空間
        optim="paged_adamw_8bit",
        remove_unused_columns=False,
        report_to="none"
    )

    trainer = RouterTrainer(
        model=model,
        args=args,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        callbacks=[ZeroGateMonitor(save_dir=save_dir)]
    )

    # 🌟 自動斷點偵測與續傳
    last_checkpoint = get_last_checkpoint(save_dir)
    if last_checkpoint is not None:
        print(f"\n🚀 偵測到中斷點 {last_checkpoint}，啟動狀態無損恢復訓練...")
    else:
        print("\n🚀 開始全新路由激發訓練...")

    trainer.train(resume_from_checkpoint=last_checkpoint)

    # 🌟 儲存最終結果
    final_path = os.path.join(save_dir, "final.pth")
    torch.save(model.state_dict() if not hasattr(model, 'module') else model.module.state_dict(), final_path)
    print(f"🎉 訓練完成！最終權重已儲存至 {final_path}")

if __name__ == "__main__":
    main()