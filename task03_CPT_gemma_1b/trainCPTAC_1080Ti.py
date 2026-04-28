# trainCPTAC_1080Ti.py — AGIV2 1080 Ti 專屬優化版 (450M 輕量化)
import os
import sys
import csv
import time
import torch
from transformers import AutoTokenizer, TrainingArguments, Trainer, set_seed, TrainerCallback
from transformers.trainer_utils import get_last_checkpoint
from datasets import load_dataset 

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 1080 Ti 環境優化
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
os.environ["BITSANDBYTES_NOWELCOME"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# 基於「恢復進度」需求，必須暫時跳過安全檢查以讀取舊格式的優化器狀態
os.environ["HF_SKIP_CHECK_TORCH_LOAD_SAFE"] = "True" 

from AGIV2GACT import AGIV2G 
from utils import AGIV2GForCausalLMT

RANDOM_SEED = 2026
set_seed(RANDOM_SEED)

# ==========================================
# [ 1080 Ti 專屬配置區 ] 
# ==========================================
# ==========================================
# [ 核心配置區 - 您可以在這裡調整所有參數 ]
# ==========================================
# 1. 路徑與基礎設定
MODEL_ID = "google/gemma-3-1b-it" 
DATASET_DIR = "./agiv2_stage1_1K"            
SAVE_DIR = "./agiv2_cpt_1080Ti_checkpoints"
LOG_PATH = "./agiv2_cpt_1080Ti_log.csv"
BEST_ROUTER_32K_PATH = "./agiv2_zerogate_ac_checkpoints_32KST/best_model.pth"
RANDOM_SEED = 42

# 2. 模型架構 (目標 ~450M 參數)
D_MODEL = 768
H_DIM = 3072
N_BLOCKS = 12
CHUNK = 256

# 3. LISA 演算法設定
LISA_N_ACTIVE = 2      # 每次同時更新的 Block 數量 (推薦 15% 左右，即 2 層)
LISA_INTERVAL = 100    # 每隔多少步重新隨機抽樣一次

# 4. 訓練超參數
MAX_STEPS = 167812                   
WARMUP_STEPS = 500                 
EVAL_STEPS = 100                   
SAVE_STEPS = 100                   
SAVE_TOTAL_LIMIT = 2               
LOGGING_STEPS = 1                  
BATCH_SIZE_PER_DEVICE = 2          
GRAD_ACCUMULATION_STEPS = 16       
LEARNING_RATE = 2e-4               
# ==========================================

# ==========================================
# [ 監控器 ]
# ==========================================
class QuantumCPTMonitor(TrainerCallback):
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
                        if len(row) > 2 and row[2].strip(): 
                            try:
                                val = float(row[2])
                                if val < self.best_eval_loss: self.best_eval_loss = val
                            except ValueError: pass
                if self.best_eval_loss != float('inf'):
                    print(f"\n📈 [Monitor] 當前最佳 Eval Loss 基準: {self.best_eval_loss:.4f}")
            except Exception as e:
                print(f"\n⚠️ [Monitor] 讀取歷史紀錄失敗: {e}")
        else:
            with open(self.path, 'w', newline='') as f:
                csv.writer(f).writerow(['step', 'loss', 'eval_loss', 'g_loc', 'g_mem', 'g_fft', 'time'])

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
            
            # 更新 logs 結構以供後續 callback 使用
            logs["g_loc"] = round(g_loc_avg, 4)
            logs["g_mem"] = round(g_mem_avg, 4)
            logs["g_fft"] = round(g_fft_avg, 4)
            
            with open(self.path, 'a', newline='') as f:
                csv.writer(f).writerow([
                    state.global_step, 
                    logs.get("loss", ""), 
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
                    best_model_path = os.path.join(self.save_dir, "best_cpt_model.safetensors")
                    raw_model = model.module if hasattr(model, 'module') else model
                    from safetensors.torch import save_model
                    save_model(raw_model, best_model_path)
                    print(f"\n[Monitor] 🌟 收斂突破 ({old_best:.4f} -> {current_eval_loss:.4f})，儲存至 {best_model_path}")
            else:
                print(f"\n[Monitor] 🛡️ 此次成績 ({current_eval_loss:.4f}) 未超越歷史最佳 ({self.best_eval_loss:.4f})。")

# ==========================================
# [ LISA (Layerwise Importance Sampling) 控制器 ]
# ==========================================
import random

class LISATrainingCallback(TrainerCallback):
    def __init__(self, n_blocks=12, n_active=2, interval=100):
        self.n_blocks = n_blocks
        self.n_active = n_active
        self.interval = interval
        self.last_step = -1

    def on_step_begin(self, args, state, control, **kwargs):
        step = state.global_step
        # 每隔 interval 步執行一次隨機抽樣
        if step % self.interval == 0 and step != self.last_step:
            self.last_step = step
            model = kwargs.get('model', None)
            if model is not None:
                self.apply_lisa_sampling(model)

    def apply_lisa_sampling(self, model):
        raw_model = model.module if hasattr(model, 'module') else model
        core_model = raw_model.base_model if hasattr(raw_model, 'base_model') else raw_model
        
        # 1. 先全部凍結
        for param in model.parameters():
            param.requires_grad = False
            
        # 2. 解凍常駐核心 (LISA 建議保留 Embedding 與 Head 以維持基本穩定)
        if hasattr(core_model, 'embed'):
            for param in core_model.embed.parameters(): param.requires_grad = True
        if hasattr(raw_model, 'lm_head'):
            for param in raw_model.lm_head.parameters(): param.requires_grad = True

        # 3. 隨機抽取 n_active 個 Blocks 進行「外科手術級更新」
        all_indices = list(range(self.n_blocks))
        active_indices = random.sample(all_indices, self.n_active)
        
        active_indices.sort()
        active_str = ", ".join([str(i) for i in active_indices])
        print(f"\n💉 [LISA] 執行隨機層採樣更新：選中 Blocks [{active_str}]")

        for i in active_indices:
            for param in core_model.blocks[i].parameters():
                param.requires_grad = True

# ==========================================
# [ 數據整理 ]
# ==========================================
class CPTDataCollator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id
        
    def __call__(self, features):
        batch_input_ids, batch_labels, batch_n_splits = [], [], []
        max_len = max(len(f["input_ids"]) for f in features)
        for f in features:
            seq = f["input_ids"]
            n_split = min(f["n_split_index"], len(seq)) 
            pad_len = max_len - len(seq)
            batch_input_ids.append(seq + [self.pad_token_id] * pad_len)
            batch_labels.append([-100] * n_split + seq[n_split:] + [-100] * pad_len)
            batch_n_splits.append(n_split)
        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
            "n_split_index": torch.tensor(batch_n_splits, dtype=torch.long)
        }

# ==========================================
# [ 核心引擎 ]
# ==========================================
def main():
    print(f"\n🚀 啟動 AGIV2 1080 Ti 優化矩陣 (D={D_MODEL}, Blocks={N_BLOCKS})...")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    parquet_file = os.path.join(DATASET_DIR, "agiv2_stage1_N_B.parquet")
    if not os.path.exists(parquet_file):
        print(f"❌ 找不到 Parquet 資料: {parquet_file}")
        return

    dataset_dict = load_dataset("parquet", data_files=parquet_file)
    
    dataset = dataset_dict['train'].train_test_split(test_size=50, seed=RANDOM_SEED)
    train_ds, eval_ds = dataset['train'], dataset['test']
    print(f"✅ 資料映射與絕對隔離完成！訓練集: {len(train_ds):,} 筆 | 獨立驗證集: {len(eval_ds)} 筆")

    # 實例化下修型模型
    base = AGIV2G(
        vocab_size=tokenizer.vocab_size, 
        D=D_MODEL, 
        C=CHUNK, 
        hidden_dim=H_DIM, 
        num_blocks=N_BLOCKS
    )
    model = AGIV2GForCausalLMT(base, use_gc=True)
    print("✅ 模型實例化完成，採用原生隨機初始化。")
    
    if os.path.exists(BEST_ROUTER_32K_PATH):
        print(f"\n🔄 偵測到基礎權重: {BEST_ROUTER_32K_PATH}")
        if BEST_ROUTER_32K_PATH.endswith(".safetensors"):
            from safetensors.torch import load_model
            # load_model 直接將權重加載到模型，不產生 state_dict 副本，極度節省 RAM
            load_model(model, BEST_ROUTER_32K_PATH, strict=False)
            print("✅ Safetensors 路由權重載入完成 (省記憶體模式)。")
        else:
            print("⚠️ 權重檔案非 Safetensors 格式，基於安全考量跳過載入。")
    else:
        print(f"\n⚠️ 找不到 32K 路由權重檔案 {BEST_ROUTER_32K_PATH}，將完全從零開始訓練！")

    # 初始凍結 (由 Callback 接管)
    print("\n🔧 啟動 LISA 演算法 (外科手術級隨機層更新模式)...")
            
    # 1080 Ti 使用 mixed precision (由 fp16=True 處理)
    model = model.cuda()

    args = TrainingArguments(
        output_dir=SAVE_DIR,
        max_steps=MAX_STEPS,                    
        per_device_train_batch_size=BATCH_SIZE_PER_DEVICE,
        per_device_eval_batch_size=BATCH_SIZE_PER_DEVICE,       
        gradient_accumulation_steps=GRAD_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_steps=WARMUP_STEPS,              
        fp16=True, # 1080 Ti 必選 fp16
        logging_steps=LOGGING_STEPS,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        optim="paged_adamw_8bit",       # 回歸最穩定且快速的 8-bit 優化器
        remove_unused_columns=False,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=CPTDataCollator(pad_id),   
        callbacks=[
            QuantumCPTMonitor(path=LOG_PATH, save_dir=SAVE_DIR),
            LISATrainingCallback(n_blocks=N_BLOCKS, n_active=LISA_N_ACTIVE, interval=LISA_INTERVAL)
        ]
    )

    last_checkpoint = get_last_checkpoint(SAVE_DIR)
    if last_checkpoint:
        print(f"🔄 偵測到舊進度: {last_checkpoint}")
        weight_path = os.path.join(last_checkpoint, "model.safetensors")
        if not os.path.exists(weight_path):
             weight_path = os.path.join(last_checkpoint, "pytorch_model.bin")
        
        if os.path.exists(weight_path):
            print(f"📦 正在僅恢復權重以避開 OOM: {weight_path}")
            if weight_path.endswith(".safetensors"):
                from safetensors.torch import load_model
                load_model(model, weight_path, strict=False)
            else:
                model.load_state_dict(torch.load(weight_path, map_location="cpu"), strict=False)
            print("✅ 權重恢復完成。正在開啟全新 8-bit 優化器...")
    
    # 強制不載入 optimizer.pt，避開 1080 Ti 的顯存壓力
    trainer.train(resume_from_checkpoint=None)
    
    final_path = os.path.join(SAVE_DIR, "final_agiv2_cpt_1080ti.safetensors")
    from safetensors.torch import save_model
    save_model(model if not hasattr(model, 'module') else model.module, final_path)
    print(f"✅ 訓練完成，模型已安全儲存至: {final_path}")

if __name__ == "__main__":
    main()
