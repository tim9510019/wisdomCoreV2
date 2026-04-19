"""
trainRouterAC.py — 物理隔離對齊終極版 (N -> N+B 範式)
===========================================================
對齊項目：
1. 具備歷史回溯與拒絕覆蓋提示的 Monitor (完全對齊 trainRouter.py)。
2. 全自動斷點續傳：自動偵測 save_dir 中的 checkpoint。
3. 儲存機制對齊：save_steps=10, save_total_limit=2。
4. 物理隔離：針對 AGIV2GAC 雙向架構，物理抹除 B 區域，且 Loss 只計算真實答案 (忽略 PAD)。
"""
import os
import sys
import csv
import time
import torch
from transformers import AutoTokenizer, TrainingArguments, Trainer, set_seed, TrainerCallback
from transformers.trainer_utils import get_last_checkpoint
from datasets import Dataset

# 確保路徑正確
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

os.environ["CUDA_VISIBLE_DEVICES"] = "1" 
os.environ["BITSANDBYTES_NOWELCOME"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from AGIV2GAC import AGIV2G 
from utils import AGIV2GForCausalLM, transplant_and_freeze, QuantumRouterEngineAC, DynamicACDataset

set_seed(2026)

# ==========================================
# 0. 訓練配置核心參數區 (方便統一修改)
# ==========================================
MODEL_ID = "google/gemma-3-1b-it"
SAVE_DIR = "./agiv2_zerogate_ac_checkpoints_1K"
LOG_PATH = "./zerogate_32k_ac_1k_log.csv"

# 🚀 訓練步數與排程 (使用者自訂)
MAX_STEPS = 2000                  # 🏆 指定要訓練的總步數 (取代 num_train_epochs)
WARMUP_STEPS = 200                 # 🌟 改用 WARMUP_STEPS 避免 deprecated 警告 (約 10%)
EVAL_STEPS = 50                    # 多少步進行一次評估
SAVE_STEPS = 10                    # 多少步保存一次 Checkpoint
SAVE_TOTAL_LIMIT = 2               # 最大保留 Checkpoint 數量
LOGGING_STEPS = 1                  # 多少步記錄一次 Log

# 🚀 模型與資料參數ㄡ
TARGET_LENGTHS = [1024]            # 訓練資料基準長度
B_SIZE = 512                       # 隔離預測區塊 (B Block) 尺寸
NUM_TRAIN_SAMPLES = 200000         # 訓練集動態生成池大小
NUM_EVAL_SAMPLES = 50              # 評估集動態生成總數
BATCH_SIZE_PER_DEVICE = 1          # 單卡 Batch Size
GRAD_ACCUMULATION_STEPS = 32       # 梯度累積步數
LEARNING_RATE = 3e-4               # 學習率


# ==========================================
# 1. 具備記憶能力的監控器
# ==========================================
class ZeroGateMonitor(TrainerCallback):
    def __init__(self, path=LOG_PATH, save_dir=SAVE_DIR):
        self.path = path
        self.save_dir = save_dir
        self.best_eval_loss = float('inf')
        os.makedirs(self.save_dir, exist_ok=True)
        
        # 🌟 歷史回溯：掃描 CSV 找回之前的最佳成績
        if os.path.exists(self.path):
            try:
                with open(self.path, 'r', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    next(reader, None) 
                    for row in reader:
                        if len(row) > 2 and row[2].strip(): 
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
# 2. 物理隔離 Trainer 與 資料收集器 (DataCollator)
# ==========================================
class ACDataCollator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id
        
    def __call__(self, features):
        max_pos = max(len(f["pos_ids"]) for f in features)
        max_neg = max(len(f["neg_ids"]) for f in features)
        
        # 🌟 使用 Left Padding (左側填充)
        # 這樣可確保尾部的 B_SIZE Tokens 絕對對齊，不破壞 N -> B 的物理隔離範圍
        for f in features:
            f["pos_ids"] = [self.pad_token_id] * (max_pos - len(f["pos_ids"])) + list(f["pos_ids"])
            f["neg_ids"] = [self.pad_token_id] * (max_neg - len(f["neg_ids"])) + list(f["neg_ids"])
            
        return {
            "pos_ids": torch.tensor([f["pos_ids"] for f in features], dtype=torch.long),
            "neg_ids": torch.tensor([f["neg_ids"] for f in features], dtype=torch.long)
        }


class RouterTrainerAC(Trainer):
    def __init__(self, pad_token_id=0, b_size=512, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pad_token_id = pad_token_id
        self.b_size = b_size

    def _physical_isolation_forward(self, model, input_ids):
        # 嚴格的邊界切分：前 N 為已知，後 B 為待預測
        L = input_ids.shape[1]
        N_len = L - self.b_size
        
        # 1. 輸入隔離：抹除 B 區塊
        inputs_isolated = input_ids.clone()
        inputs_isolated[:, N_len:] = self.pad_token_id 
        
        # 2. 標籤隔離：前 N 不算分 (-100)
        labels = torch.full_like(input_ids, -100)
        labels[:, N_len:] = input_ids[:, N_len:] 
        
        # 3. 極致對焦：在 B 區塊中，忽略所有的 PAD，火力全開針對真實 Token
        labels[labels == self.pad_token_id] = -100
        
        return model(input_ids=inputs_isolated, labels=labels)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        pos_ids, neg_ids = inputs.get("pos_ids"), inputs.get("neg_ids")
        
        with torch.no_grad():
            neg_loss = self._physical_isolation_forward(model, neg_ids)["loss"].detach()
        
        pos_outputs = self._physical_isolation_forward(model, pos_ids)
        pos_loss = pos_outputs["loss"]
        
        raw_model = model.module if hasattr(model, 'module') else model
        core_model = raw_model.base_model if hasattr(raw_model, 'base_model') else raw_model
        
        gate_penalty = 0.0
        for b in core_model.blocks:
            if hasattr(b, 'gate_mem'):
                # 🌟 將此處的 20.0 降低為 15.0 以適應 4K 訓練
                gate_penalty = gate_penalty + torch.exp(-torch.abs(b.gate_mem) * 15.0).mean()
        
        # 三元對抗 Loss 計算保持不變
        contrastive_loss = torch.clamp(pos_loss - neg_loss + 1.0, min=0.0).mean()
        total_loss = pos_loss + 0.5 * contrastive_loss + 0.1 * gate_penalty
        return (total_loss.mean(), pos_outputs) if return_outputs else total_loss.mean()

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            loss = self.compute_loss(model, inputs, return_outputs=False)
        if prediction_loss_only: return (loss, None, None)
        
        pos_ids = inputs.get("pos_ids")
        N_len = pos_ids.shape[1] - self.b_size
        
        # 評估階段同樣執行嚴格物理隔離
        inputs_iso = pos_ids.clone()
        inputs_iso[:, N_len:] = self.pad_token_id
        with torch.no_grad():
            # 這裡傳入完整的 pos_ids 作為 labels，模型會產出對應的 Logits 供後續 metrics 計算
            outputs = model(input_ids=inputs_iso, labels=pos_ids)
            
        return (loss, outputs.logits, pos_ids)

# ==========================================
# 3. 主流程
# ==========================================
def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    print("\n📦 [資料集] 正在初始化及時數據生成引擎...")
    engine = QuantumRouterEngineAC()

    train_ds = DynamicACDataset(engine, TARGET_LENGTHS, NUM_TRAIN_SAMPLES)
    eval_ds = DynamicACDataset(engine, TARGET_LENGTHS, NUM_EVAL_SAMPLES)

    # Transformer Chunk 256 Grok 比較快
    base = AGIV2G(vocab_size=262144, D=1152, C=256, hidden_dim=6912, num_blocks=26)
    base = transplant_and_freeze(MODEL_ID, base)
    model = AGIV2GForCausalLM(base, use_gc=True)
    
    unlock_keywords = ["gate_fft", "gate_mem", "omegas", "mlp_H", "fft_norm", "Q_mem", "W_k_mem", "W_v_mem", "mem_norm", "W_q_cross", "o_proj_cross"]
    for name, param in model.named_parameters():
        param.requires_grad = any(k in name for k in unlock_keywords)
        if "gate_fft" in name or "gate_mem" in name:
            param.register_hook(lambda grad: grad * 10.0)
            
    model = model.cuda().to(torch.bfloat16)

    # 🌟 完全對齊 TrainingArguments 
    args = TrainingArguments(
        output_dir=SAVE_DIR,
        max_steps=MAX_STEPS,                    # 🌟 替換 num_train_epochs，由上面指定
        per_device_train_batch_size=BATCH_SIZE_PER_DEVICE,
        per_device_eval_batch_size=BATCH_SIZE_PER_DEVICE,       
        gradient_accumulation_steps=GRAD_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_steps=WARMUP_STEPS,              # 🌟 取代 warmup_ratio，解決 warnings
        bf16=True,
        logging_steps=LOGGING_STEPS,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        optim="paged_adamw_8bit",
        remove_unused_columns=False,
        report_to="none"
    )

    trainer = RouterTrainerAC(
        pad_token_id=pad_id,
        b_size=B_SIZE,
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=ACDataCollator(pad_id),   # 🌟 修正 ValueError 的關鍵，提供自訂的 DataCollator 加入 Left Padding
        callbacks=[ZeroGateMonitor(path=LOG_PATH, save_dir=SAVE_DIR)]
    )

    # 🌟 自動斷點偵測
    last_checkpoint = get_last_checkpoint(SAVE_DIR)
    if last_checkpoint is not None:
        print(f"\n🚀 偵測到中斷點 {last_checkpoint}，啟動狀態無損恢復訓練...")
    else:
        print("\n🚀 開始全新物理隔離訓練 (N -> N+B)...")

    trainer.train(resume_from_checkpoint=last_checkpoint)

    # 🌟 儲存最終結果
    final_path = os.path.join(SAVE_DIR, "final_router_ac.pth")
    torch.save(model.state_dict() if not hasattr(model, 'module') else model.module.state_dict(), final_path)
    print(f"🎉 訓練完成！最終權重已儲存至 {final_path}")

if __name__ == "__main__":
    main()