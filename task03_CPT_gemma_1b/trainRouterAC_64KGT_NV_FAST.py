# trainRouterAC_64KGT_NV_FAST.py — 物理隔離對齊終極版 (N -> N+B 範式 + GT 動態 Router + NV 雙卡配置 + Fusion 終極加速)
import os
import sys
import csv
import time
import threading
import torch
from huggingface_hub import HfApi
from transformers import AutoTokenizer, TrainingArguments, Trainer, set_seed, TrainerCallback
from transformers.trainer_utils import get_last_checkpoint
from datasets import Dataset

# 確保路徑正確
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
os.environ["BITSANDBYTES_NOWELCOME"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# 🌟 改為引入加速版 AGIV2GFast
from AGIV2G_Fast import AGIV2GFast 
from utils import AGIV2GForCausalLMT, transplant_and_freeze, QuantumRouterEngineAC, DynamicACDataset

set_seed(2026)

# ==========================================
# 0. 訓練配置核心參數區 (方便統一修改)
# ==========================================
MODEL_ID = "google/gemma-3-1b-it"
SAVE_DIR = "~/agiv2_zerogate_ac_checkpoints_64KGT_NV_FAST"
LOG_PATH = "~/zerogate_32k_ac_64kgt_nv_fast_log.csv"
# 一樣可延續先前的 32K 基礎
BEST_32K_PATH = "~/agiv2_zerogate_ac_checkpoints_32KGT_NV/best_model.pth"

SAVE_DIR = os.path.expanduser(SAVE_DIR)
LOG_PATH = os.path.expanduser(LOG_PATH)
BEST_32K_PATH = os.path.expanduser(BEST_32K_PATH)

# 🚀 訓練步數與排程 (使用者自訂)
MAX_STEPS = 2000                   
WARMUP_STEPS = 200                 
EVAL_STEPS = 50                    
SAVE_STEPS = 10                    
SAVE_TOTAL_LIMIT = 2               
LOGGING_STEPS = 1                  

# 🚀 模型與資料參數
TARGET_LENGTHS = [65536]            
B_SIZE = 512                       
NUM_TRAIN_SAMPLES = 200000         
NUM_EVAL_SAMPLES = 50              
BATCH_SIZE_PER_DEVICE = 1          
GRAD_ACCUMULATION_STEPS = 32       
LEARNING_RATE = 1e-5               
CHUNK = 256
GATE_ENTROPY_LAMBDA = 0.01
ROPE_LOCAL = 10000.0
ROPE_GLOBAL = 1000000.0

# HuggingFace 自動上傳配置
REPO_ID = "tim9510019/AGIV2-1B-Router_64KGT_NV_FAST"
HF_CE_LOSS_THRESHOLD = 3.9

# ==========================================
# [ 硬體保護：主動式散熱控制器 ]
# ==========================================
class ThermalControlCallback(TrainerCallback):
    def __init__(self, delay_seconds=1.5):
        self.delay_seconds = delay_seconds

    def on_step_end(self, args, state, control, **kwargs):
        time.sleep(self.delay_seconds)

# ==========================================
# 1. 具備記憶能力的監控器 (整合 GT 觀測)
# ==========================================
class QuantumRouterMonitor(TrainerCallback):
    def __init__(self, path=LOG_PATH, save_dir=SAVE_DIR):
        self.path = path
        self.save_dir = save_dir
        self.best_eval_loss = float('inf')
        os.makedirs(self.save_dir, exist_ok=True)
        
        if os.path.exists(self.path):
            try:
                with open(self.path, 'r', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    next(reader, None) 
                    for row in reader:
                        if len(row) > 4 and row[4].strip(): 
                            try:
                                val = float(row[4])
                                if val < self.best_eval_loss:
                                    self.best_eval_loss = val
                            except ValueError: pass
                if self.best_eval_loss != float('inf'):
                    print(f"\n📈 [Monitor] 成功找回歷史紀錄！當前最佳 Eval Loss 基準: {self.best_eval_loss:.4f}")
            except Exception as e:
                print(f"\n⚠️ [Monitor] 讀取歷史紀錄失敗: {e}")
        else:
            with open(self.path, 'w', newline='') as f:
                csv.writer(f).writerow(['step', 'loss', 'ce_loss', 'gate_ent_loss', 'eval_loss', 'g_loc', 'g_mem', 'g_fft', 'time'])

    def on_step_begin(self, args, state, control, **kwargs):
        model = kwargs.get('model', None)
        if model is not None:
            progress = state.global_step / max(1, state.max_steps)
            current_temp = max(0.5, 2.0 - 1.5 * progress)
            
            raw_model = model.module if hasattr(model, 'module') else model
            core_model = raw_model.base_model if hasattr(raw_model, 'base_model') else raw_model
            if hasattr(core_model, 'set_temperature'):
                core_model.set_temperature(current_temp)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            model = kwargs.get('model', None)
            g_loc_avg, g_mem_avg, g_fft_avg = 0.0, 0.0, 0.0
            blocks_count = 0
            
            if model is not None:
                raw_model = model.module if hasattr(model, 'module') else model
                core_model = raw_model.base_model if hasattr(raw_model, 'base_model') else raw_model
                
                for b in core_model.blocks:
                    if hasattr(b, 'avg_g_loc'):
                        g_loc_avg += b.avg_g_loc.item()
                        g_mem_avg += b.avg_g_mem.item()
                        g_fft_avg += b.avg_g_fft.item()
                        blocks_count += 1
                        
            if blocks_count > 0:
                g_loc_avg /= blocks_count
                g_mem_avg /= blocks_count
                g_fft_avg /= blocks_count
            
            logs["g_loc"] = round(g_loc_avg, 4)
            logs["g_mem"] = round(g_mem_avg, 4)
            logs["g_fft"] = round(g_fft_avg, 4)
            
            ce_loss_val = ""
            gate_ent_val = ""
            if model is not None:
                raw_model = model.module if hasattr(model, 'module') else model
                if hasattr(raw_model, '_last_ce_loss'):
                    ce_loss_val = f"{raw_model._last_ce_loss:.6f}"
                    logs["ce_loss"] = raw_model._last_ce_loss
                if hasattr(raw_model, '_last_gate_entropy_loss'):
                    gate_ent_val = f"{raw_model._last_gate_entropy_loss:.6f}"
                    logs["gate_ent_loss"] = raw_model._last_gate_entropy_loss
            
            with open(self.path, 'a', newline='') as f:
                csv.writer(f).writerow([
                    state.global_step, 
                    logs.get("loss", ""), 
                    ce_loss_val,
                    gate_ent_val,
                    logs.get("eval_loss", ""), 
                    f"{g_loc_avg:.4f}", 
                    f"{g_mem_avg:.4f}", 
                    f"{g_fft_avg:.4f}", 
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
                    raw_model = model.module if hasattr(model, 'module') else model
                    torch.save(raw_model.state_dict(), best_model_path)
                    print(f"\n[Monitor] 🌟 發現更佳權重 ({old_best:.4f} -> {current_eval_loss:.4f})，已儲存至 {best_model_path}")
            else:
                print(f"\n[Monitor] 🛡️ 此次成績 ({current_eval_loss:.4f}) 未超越歷史最佳 ({self.best_eval_loss:.4f})，跳過保存。")

# ==========================================
# [ HuggingFace 自動上傳器 ]
# ==========================================
class HFAutoUploadCallback(TrainerCallback):
    def __init__(self, repo_id=REPO_ID, model_path=None, log_path=LOG_PATH,
                 ce_loss_threshold=HF_CE_LOSS_THRESHOLD, save_dir=SAVE_DIR):
        self.repo_id = repo_id
        self.model_path = model_path or os.path.join(save_dir, "best_model.pth")
        self.log_path = log_path
        self.ce_loss_threshold = ce_loss_threshold
        self.best_ce_loss = float('inf')
        self._upload_lock = threading.Lock()   

    def _do_upload(self, ce_loss_val, step):
        with self._upload_lock:
            try:
                api = HfApi()
                api.create_repo(repo_id=self.repo_id, exist_ok=True, repo_type="model")

                if os.path.isfile(self.model_path):
                    api.upload_file(
                        path_or_fileobj=self.model_path,
                        path_in_repo=os.path.basename(self.model_path),
                        repo_id=self.repo_id,
                        repo_type="model",
                        commit_message=f"[Auto] step={step} ce_loss={ce_loss_val:.6f}"
                    )
                    print(f"\n[HF Upload] ✅ Best Model 上傳完成 (step={step}, ce_loss={ce_loss_val:.4f})")
                else:
                    print(f"\n[HF Upload] ⚠️ 找不到模型檔案 {self.model_path}，跳過模型上傳。")

                if os.path.isfile(self.log_path):
                    api.upload_file(
                        path_or_fileobj=self.log_path,
                        path_in_repo=os.path.basename(self.log_path),
                        repo_id=self.repo_id,
                        repo_type="model",
                        commit_message=f"[Auto] log update step={step}"
                    )
                    print(f"[HF Upload] ✅ CSV Log 上傳完成")
                else:
                    print(f"[HF Upload] ⚠️ 找不到 CSV Log {self.log_path}，跳過 Log 上傳。")

                print(f"[HF Upload] 🔗 https://huggingface.co/{self.repo_id}/tree/main")
            except Exception as e:
                print(f"\n[HF Upload] ❌ 上傳失敗: {e}")

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None:
            return

        ce_loss = metrics.get("eval_loss", None)
        if ce_loss is None:
            return

        if ce_loss < self.best_ce_loss and ce_loss < self.ce_loss_threshold:
            self.best_ce_loss = ce_loss
            step = state.global_step
            print(f"\n[HF Upload] 🚀 CE Loss 突破門檻 ({ce_loss:.4f} < {self.ce_loss_threshold})，啟動背景上傳...")
            t = threading.Thread(
                target=self._do_upload,
                args=(ce_loss, step),
                daemon=True
            )
            t.start()

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
        L = input_ids.shape[1]
        N_len = L - self.b_size
        
        inputs_isolated = input_ids.clone()
        inputs_isolated[:, N_len:] = self.pad_token_id 
        
        labels = torch.full_like(input_ids, -100)
        labels[:, N_len:] = input_ids[:, N_len:] 
        
        labels[labels == self.pad_token_id] = -100
        
        return model(input_ids=inputs_isolated, labels=labels)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        pos_ids, neg_ids = inputs.get("pos_ids"), inputs.get("neg_ids")
        
        with torch.no_grad():
            neg_loss = self._physical_isolation_forward(model, neg_ids)["loss"].detach()
        
        pos_outputs = self._physical_isolation_forward(model, pos_ids)
        pos_loss = pos_outputs["loss"]
        
        # 三元對抗 Loss 計算保持不變，但移除舊版閘門懲罰 (交由 LMT 處理)
        contrastive_loss = torch.clamp(pos_loss - neg_loss + 1.0, min=0.0).mean()
        total_loss = pos_loss + 0.5 * contrastive_loss
        
        return (total_loss.mean(), pos_outputs) if return_outputs else total_loss.mean()

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            loss = self.compute_loss(model, inputs, return_outputs=False)
        if prediction_loss_only: return (loss, None, None)
        
        pos_ids = inputs.get("pos_ids")
        N_len = pos_ids.shape[1] - self.b_size
        
        inputs_iso = pos_ids.clone()
        inputs_iso[:, N_len:] = self.pad_token_id
        with torch.no_grad():
            outputs = model(input_ids=inputs_iso, labels=pos_ids)
            
        return (loss, outputs.logits, pos_ids)

# ==========================================
# 3. 主流程
# ==========================================
def main():
    print("\n🚀 啟動 AGIV2 Router 64K 訓練矩陣 (FUSION 終極加速版 + 64K GT)...")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    print("\n📦 [資料集] 正在初始化及時數據生成引擎...")
    engine = QuantumRouterEngineAC()

    train_ds = DynamicACDataset(engine, TARGET_LENGTHS, NUM_TRAIN_SAMPLES)
    eval_ds = DynamicACDataset(engine, TARGET_LENGTHS, NUM_EVAL_SAMPLES)

    # 🌟 實例化加速版的 AGIV2GFast
    base = AGIV2GFast(vocab_size=262144, D=1152, C=CHUNK, hidden_dim=6912, num_blocks=26, rope_local=ROPE_LOCAL, rope_global=ROPE_GLOBAL)
    
    print("\n🔄 正在掛載原生 google/gemma-3-1b-it 權重...")
    base = transplant_and_freeze(MODEL_ID, base)
    
    model = AGIV2GForCausalLMT(base, use_gc=True, gate_entropy_lambda=GATE_ENTROPY_LAMBDA)
    print("✅ 模型實例化完成，並成功接合官方預訓練特徵。")
    
    # 🌟 載入 32K 階段的 best_model 權重
    if os.path.exists(BEST_32K_PATH):
        print(f"\n🔄 成功尋獲 32K 階段之最佳權重，正在載入以進行 64K 延伸訓練: {BEST_32K_PATH}")
        state_dict = torch.load(BEST_32K_PATH, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
    else:
        print(f"\n⚠️ 找不到 32K 權重檔案 {BEST_32K_PATH}，將從頭開始訓練！")
        
    # 🌟 觸發 Level 1 QKV 融合 (必須在所有權重載入之後執行！)
    if hasattr(base, 'fuse_qkv'):
        base.fuse_qkv()
    
    print("\n🔧 啟動全參數訓練設定 (包含動態 Router):")
    for param in model.parameters():
        param.requires_grad = True
        
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✅ 設定完成。可訓練參數總量: {trainable_params:,} (100% 全開)")

    model = model.cuda().to(torch.bfloat16)

    args = TrainingArguments(
        output_dir=SAVE_DIR,
        max_steps=MAX_STEPS,                    
        per_device_train_batch_size=BATCH_SIZE_PER_DEVICE,
        per_device_eval_batch_size=BATCH_SIZE_PER_DEVICE,       
        gradient_accumulation_steps=GRAD_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_steps=WARMUP_STEPS,              
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
        data_collator=ACDataCollator(pad_id),   
        callbacks=[
            QuantumRouterMonitor(path=LOG_PATH, save_dir=SAVE_DIR),
            ThermalControlCallback(delay_seconds=0.0),
            HFAutoUploadCallback()
        ]
    )

    last_checkpoint = get_last_checkpoint(SAVE_DIR)
    if last_checkpoint is not None:
        print(f"\n🚀 偵測到中斷點 {last_checkpoint}，啟動狀態無損恢復訓練...")
        trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        print("\n🚀 開始全新物理隔離訓練 (N -> N+B)...")
        trainer.train()

    final_path = os.path.join(SAVE_DIR, "final_router_ac_64kgt_fast.pth")
    torch.save(model.state_dict() if not hasattr(model, 'module') else model.module.state_dict(), final_path)
    print(f"🎉 訓練完成！最終權重已儲存至 {final_path}")

if __name__ == "__main__":
    main()
