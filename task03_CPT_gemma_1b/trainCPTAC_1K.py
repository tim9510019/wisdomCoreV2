"""
trainCPTAC_1K.py — AGIV2 持續預訓練終極引擎 (N -> N+B 物理隔離直讀版)
===========================================================
第一性原理實踐：
1. 降維直讀：捨棄 load_from_disk 的高階封裝，直接以 Parquet 原生格式掛載實體矩陣，邊際效用最大化。
2. 實體斷層隔離：DataCollator 根據 n_split_index 將 N 的 Labels 設為 -100，
   確保梯度僅從 B (嚴格預測區) 流回，徹底封殺盲目試錯。
3. 外科手術解凍：精準解鎖 Routing + LayerNorm，凍結 LM Head 與 Attn Proj。
4. 跨階段繼承：自動載入 32K Router 階段的 Best Model，繼承已喚醒的路由能力。
5. 驗證集極小化：精準切割 50 筆絕對獨立序列，確保 Eval 階段不浪費任何算力。
"""
import os
import sys
import csv
import time
import torch
from transformers import AutoTokenizer, TrainingArguments, Trainer, set_seed, TrainerCallback
from transformers.trainer_utils import get_last_checkpoint
from datasets import load_dataset 

# 確保外部依賴路徑暢通
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 算力池綁定
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
os.environ["BITSANDBYTES_NOWELCOME"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from AGIV2GAC import AGIV2G 
from utils import AGIV2GForCausalLM, transplant_and_freeze

RANDOM_SEED = 2026

set_seed(RANDOM_SEED)

# ==========================================
# [ 全局配置區 ] 能量分佈與物理參數
# ==========================================
MODEL_ID = "google/gemma-3-1b-it"
DATASET_DIR = "./agiv2_stage1_1K"            
SAVE_DIR = "./agiv2_cpt_checkpoints_1K"
LOG_PATH = "./agiv2_cpt_1k_log.csv"

# 🌟 跨階段拓撲繼承：指向 32K 路由對齊階段的最佳權重
BEST_ROUTER_32K_PATH = "./agiv2_zerogate_ac_checkpoints_32K/best_model.pth"

# 依據 CPT 階段量化指標設定
MAX_STEPS = 5000                   
WARMUP_STEPS = 500                 
EVAL_STEPS = 100                   
SAVE_STEPS = 100                   
SAVE_TOTAL_LIMIT = 3               
LOGGING_STEPS = 1                  

BATCH_SIZE_PER_DEVICE = 2          
GRAD_ACCUMULATION_STEPS = 16       
LEARNING_RATE = 1e-4               

# ==========================================
# [ 監控維度 ] 具備歷史回溯能力的量子觀測器
# ==========================================
class QuantumCPTMonitor(TrainerCallback):
    def __init__(self, path=LOG_PATH, save_dir=SAVE_DIR):
        self.path = path
        self.save_dir = save_dir
        self.best_loss = float('inf')
        os.makedirs(self.save_dir, exist_ok=True)
        
        if os.path.exists(self.path):
            try:
                with open(self.path, 'r', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    next(reader, None) 
                    for row in reader:
                        if len(row) > 1 and row[1].strip(): 
                            try:
                                val = float(row[1])
                                if val < self.best_loss:
                                    self.best_loss = val
                            except ValueError: pass
                if self.best_loss != float('inf'):
                    print(f"\n📈 [Monitor] 時間線同步成功。當前最佳 Training Loss 基準: {self.best_loss:.4f}")
            except Exception as e:
                print(f"\n⚠️ [Monitor] 讀取歷史紀錄失敗: {e}")
        else:
            with open(self.path, 'w', newline='') as f:
                csv.writer(f).writerow(['step', 'loss', 'fft_max', 'mem_max', 'time'])

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
            
            current_loss = logs.get("loss", None)
            with open(self.path, 'a', newline='') as f:
                csv.writer(f).writerow([state.global_step, current_loss, max_fft, max_mem, time.ctime()])
                
            if current_loss and current_loss < self.best_loss:
                self.best_loss = current_loss
                if model is not None and state.global_step % (SAVE_STEPS // 2) == 0:
                    best_model_path = os.path.join(self.save_dir, "best_cpt_model.pth")
                    raw_model = model.module if hasattr(model, 'module') else model
                    torch.save(raw_model.state_dict(), best_model_path)
                    print(f"\n[Monitor] 🌟 結構極化收斂 ({current_loss:.4f})，實體權重已儲存。")

# ==========================================
# [ 物理隔離 ] N -> N+B 絕對斷層拼接器
# ==========================================
class CPTDataCollator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id
        
    def __call__(self, features):
        batch_input_ids = []
        batch_labels = []
        
        max_len = max(len(f["input_ids"]) for f in features)
        
        for f in features:
            seq = f["input_ids"]
            n_split = f["n_split_index"]
            pad_len = max_len - len(seq)
            
            # 填補 Input
            padded_seq = seq + [self.pad_token_id] * pad_len
            batch_input_ids.append(padded_seq)
            
            # 絕對物理隔離：確保 n_split 之前的梯度為 0 (設為 -100)
            n_split = min(n_split, len(seq)) # 防禦性邊界檢查
            labels = [-100] * n_split + seq[n_split:] + [-100] * pad_len
            batch_labels.append(labels)
            
        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long)
        }

# ==========================================
# [ 核心引擎 ] 主執行緒與外科手術解凍
# ==========================================
def main():
    print("\n🚀 啟動 AGIV2 CPT 第一階訓練矩陣 (N -> N+B 物理隔離直讀版)...")
    
    # 1. 初始化基礎元件
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    # 2. 載入超光速資料庫 (降維直接讀取 Parquet)
    parquet_file = os.path.join(DATASET_DIR, "agiv2_stage1_N_B.parquet")
    if not os.path.exists(parquet_file):
        raise FileNotFoundError(f"找不到實體矩陣 {parquet_file}，請先執行 cpt1K.py 產生資料。")
    
    print(f"📦 成功鎖定實體磁區，直接掛載 Parquet 拓撲: {parquet_file}")
    dataset_dict = load_dataset("parquet", data_files=parquet_file)
    
    # =========================================================================
    # 🌟 [ 驗證集極小化 ] 
    # 第一性原理：保留算力給主引擎。從原生結構中無放回抽出絕對獨立的 50 筆資料作為 Eval。
    # =========================================================================
    dataset = dataset_dict['train'].train_test_split(test_size=50, seed=RANDOM_SEED)
    train_ds = dataset['train']
    eval_ds = dataset['test']
    print(f"✅ 資料映射與絕對隔離完成！訓練集: {len(train_ds):,} 筆 | 獨立驗證集: {len(eval_ds)} 筆")

    # 3. 模型拓撲對齊
    base = AGIV2G(vocab_size=262144, D=1152, C=256, hidden_dim=6912, num_blocks=26)
    base = transplant_and_freeze(MODEL_ID, base)
    model = AGIV2GForCausalLM(base, use_gc=True)
    
    # =========================================================================
    # 🌟 [ 跨階段拓撲繼承 ] 
    # =========================================================================
    if os.path.exists(BEST_ROUTER_32K_PATH):
        print(f"\n🔄 成功尋獲 32K 階段之最佳路由權重，正在載入以進行 CPT 結構穩固訓練: {BEST_ROUTER_32K_PATH}")
        state_dict = torch.load(BEST_ROUTER_32K_PATH, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
        print("✅ 路由靈魂灌注完成，模型已具備 TRIDATA 階段的先驗解構能力。")
    else:
        print(f"\n⚠️ 找不到 32K 路由權重檔案 {BEST_ROUTER_32K_PATH}，將僅依賴原始 Gemma 權重從頭開始 CPT！")

    # =========================================================================
    # [ 外科手術解凍矩陣 ] (Surgical Unfreezing Matrix)
    # =========================================================================
    routing_keys = ["gate_fft", "gate_mem", "omegas", "mlp_H", "Q_mem", "W_k_mem", "W_v_mem", "W_q_cross", "o_proj_cross"]
    norm_keys = ["norm", "input_layernorm", "post_attention_layernorm"] 
    frozen_keys = ["lm_head", "o_proj"] 
    
    print("\n🔧 啟動外科手術解凍程序 (CPT Stage):")
    for name, param in model.named_parameters():
        param.requires_grad = False
        if any(k in name for k in routing_keys) or any(k in name for k in norm_keys):
            param.requires_grad = True
        if any(k in name for k in frozen_keys) and "o_proj_cross" not in name:
            param.requires_grad = False
        if ("gate_fft" in name or "gate_mem" in name) and param.requires_grad:
            param.register_hook(lambda grad: grad * 10.0)
            
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✅ 解凍完成。可訓練參數總量: {trainable_params:,}")

    model = model.cuda().to(torch.bfloat16)

    # 4. 訓練能量配置
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

    # 5. 啟動框架
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=CPTDataCollator(pad_id),   
        callbacks=[QuantumCPTMonitor(path=LOG_PATH, save_dir=SAVE_DIR)]
    )

    last_checkpoint = get_last_checkpoint(SAVE_DIR)
    if last_checkpoint is not None:
        print(f"\n🚀 偵測到時間錨點 {last_checkpoint}，啟動狀態無損恢復...")
    else:
        print("\n🚀 開始全新 CPT 物理隔離訓練 (N -> N+B)...")

    trainer.train(resume_from_checkpoint=last_checkpoint)

    # 6. 最終坍縮
    final_path = os.path.join(SAVE_DIR, "final_agiv2_cpt_1k.pth")
    torch.save(model.state_dict() if not hasattr(model, 'module') else model.module.state_dict(), final_path)
    print(f"🎉 第一階 CPT 拓撲對齊完成！最終權重已沉澱至 {final_path}")

if __name__ == "__main__":
    main()