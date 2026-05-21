# trainCPTAC_AGI_GEMMA3_1K_fftrope_scratch.py — AGI GEMMA 3 1D Depthwise Conv + Sinc Causal From-Scratch 訓練引擎 (全開放訓練)
import os
import sys
import csv
import time
import threading
import math
import types

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
os.environ["BITSANDBYTES_NOWELCOME"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import HfApi
from transformers import AutoTokenizer, TrainingArguments, Trainer, set_seed, TrainerCallback
from transformers.trainer_utils import get_last_checkpoint
from datasets import load_dataset 

# 🌟 官方 FlashAttention 導入
try:
    from flash_attn import flash_attn_func
except ImportError:
    raise ImportError("請確定已安裝 flash-attn 套件: pip install flash-attn --no-build-isolation")

from GEMMA3 import GEMMA3 
# 僅引入 AGIV2GForCausalLMT，不引入/不呼叫 transplant_and_freeze，從隨機初始化開始！
from utils import AGIV2GForCausalLMT

RANDOM_SEED = 2026
set_seed(RANDOM_SEED)

# ==========================================
# [ 全局配置區 ] 
# ==========================================
MODEL_ID = "google/gemma-3-1b-it" 
DATASET_DIR = "~/agiv2_stage1_1K"            
SAVE_DIR = "~/agigemma3_scratch_checkpoints_1K_fftrope"
LOG_PATH = "~/agigemma3_scratch_1k_fftrope_log.csv"

DATASET_DIR = os.path.expanduser(DATASET_DIR)
SAVE_DIR = os.path.expanduser(SAVE_DIR)
LOG_PATH = os.path.expanduser(LOG_PATH)

MAX_STEPS = 167812                   
WARMUP_STEPS = 500                 
EVAL_STEPS = 100                   
SAVE_STEPS = 100                   
SAVE_TOTAL_LIMIT = 2               
LOGGING_STEPS = 1                  
BATCH_SIZE_PER_DEVICE = 2          
GRAD_ACCUMULATION_STEPS = 16       
LEARNING_RATE = 1e-4               
CHUNK = 256
GATE_ENTROPY_LAMBDA = 0.01          
ROPE_LOCAL = 10000.0
ROPE_GLOBAL = 1000000.0

# HuggingFace 自動上傳配置
REPO_ID = "tim9510019/AGIGEMMA3-1B-SCRATCH_1K_fftrope"
HF_CE_LOSS_THRESHOLD = 9.9          # 從頭訓練初期 Loss 較高，將門檻放寬

# ==========================================
# [ 物理隔離：時域 Sinc Causal 卷積 RoPE 模組 ]
# ==========================================
class SincCausalRoPE(nn.Module):
    def __init__(self, num_layers=26, num_heads=4, head_dim=256, target_params=745472, W=64):
        """
        物理隔離的獨立調製模組。
        內部封裝一個長度完全精準相等的 745,472 可訓練參數張量，
        為每層分配 28,672 個參數（變形為 112 x 256 的相位調製矩陣），
        透過 1D Causal Sinc 卷積 (Depthwise Conv1d) 在時域上提取特徵並調製 RoPE 角度。
        """
        super().__init__()
        self.weight = nn.Parameter(torch.empty(target_params, dtype=torch.float32))
        # 使用常態分佈初始化
        nn.init.normal_(self.weight, mean=0.0, std=0.02)
        
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.params_per_layer = target_params // num_layers # 28,672
        self.W = W
        
    def forward(self, x, layer_idx):
        # x shape: (B, L, D) 來自於 decoder block 的 normed input，D = 1152
        B, L, D = x.shape
        device = x.device
        dtype = x.dtype
        
        # 提取該層專屬的 28,672 參數切片，重塑為 (112, 256) 相位映射矩陣
        start_idx = layer_idx * self.params_per_layer
        end_idx = start_idx + self.params_per_layer
        w_layer = self.weight[start_idx:end_idx].view(112, 256)
        
        # 強制以 float32 進行穩定投影與 Sinc 卷積核構造，避免 bfloat16 與特殊函數衝突
        with torch.autocast(device_type="cuda", enabled=False):
            x_slice = x[..., :112].to(torch.float32)
            w_layer_cast = w_layer.to(torch.float32)
            
            # 矩陣相乘投射至 256 維度的調製特徵
            h = torch.matmul(x_slice, w_layer_cast) # shape: (B, L, 256)
            
            # 構造通道獨立 (Depthwise) 的 Causal Sinc 卷積核
            t = torch.arange(self.W, device=device, dtype=torch.float32) # [W]
            t_0 = (self.W - 1) / 2.0
            
            # 通道獨立的截止頻率 fc, 在 [0.02, 0.48] 之間均勻分布 (靜態常數，不引入額外參數)
            fc = torch.linspace(0.02, 0.48, steps=256, device=device, dtype=torch.float32)
            
            # Sinc 濾波器公式: 2 * fc * sinc(2 * fc * (t - t_0))
            args = 2.0 * fc.unsqueeze(1) * (t.unsqueeze(0) - t_0)
            sinc_val = torch.sinc(args) # [256, W]
            
            # Hamming 窗
            window = 0.54 - 0.46 * torch.cos(2.0 * math.pi * t / (self.W - 1))
            window = window.unsqueeze(0) # [1, W]
            
            # 理想 Sinc 卷積核
            kernel = 2.0 * fc.unsqueeze(1) * sinc_val * window # [256, W]
            # 能量 L1 歸一化，維持訓練及梯度極限穩定
            kernel = kernel / (kernel.norm(p=1, dim=1, keepdim=True) + 1e-8)
            
            # 重塑為 Depthwise Conv1d 的濾波器權重格式: [256, 1, W]
            weight_conv = kernel.unsqueeze(1)
            
            # 執行 Causal Depthwise Conv1d (左側補零 W-1 以保證因果性)
            h_t = h.transpose(1, 2) # [B, 256, L]
            h_padded = F.pad(h_t, (self.W - 1, 0)) # [B, 256, L + W - 1]
            
            # 執行卷積互相關
            h_modulated = F.conv1d(h_padded, weight_conv, groups=256) # [B, 256, L]
            h_modulated = h_modulated.transpose(1, 2) # [B, L, 256]
            
            # 重塑與擴展至 RoPE 角度修正形狀 (B, L, num_heads, head_dim // 2) -> (B, L, 4, 128)
            h_expanded = torch.cat([h_modulated, h_modulated], dim=-1) # shape: (B, L, 512)
            h_reshaped = h_expanded.view(B, L, self.num_heads, self.head_dim // 2)
            
        return h_reshaped.to(dtype)

# ==========================================
# [ 相位鎖定 Rotary Position Embedding ]
# ==========================================
def apply_phase_locked_rope(x, head_dim, base_freq=1000000.0, delta_theta=None):
    B, L, num_heads, D = x.shape
    position = torch.arange(L, device=x.device).unsqueeze(1).float()
    div_term = torch.exp(
        torch.arange(0, head_dim, 2, device=x.device).float()
        * -(math.log(base_freq) / head_dim)
    )
    freqs = position * div_term # shape: (L, D // 2)
    emb = freqs.unsqueeze(0).unsqueeze(2) # shape: (1, L, 1, D // 2)
    
    # 注入 FFT 調製後的動態相位偏差 delta_theta
    if delta_theta is not None:
        emb = emb + delta_theta # shape: (B, L, num_heads, D // 2)
        
    emb = torch.cat((emb, emb), dim=-1) # shape: (B, L, num_heads, D)
    sin_val = torch.sin(emb).to(x.dtype)
    cos_val = torch.cos(emb).to(x.dtype)
    x1 = x[..., : head_dim // 2]
    x2 = x[..., head_dim // 2 :]
    x_rotated = torch.cat((-x2, x1), dim=-1)
    return (x * cos_val) + (x_rotated * sin_val)

# ==========================================
# [ 動態修補前向傳播：接合 FFT 相位鎖定 ]
# ==========================================
def make_phase_locked_decoder_forward(layer_idx, fft_phase_lock_module):
    def forward(self, X, shift_size=0, n_split_index=None):
        B, L, D = X.shape

        normed_X = self.input_layernorm(X)

        Q = self.W_q_loc(normed_X).view(B, L, self.num_heads, self.head_dim)
        K = self.W_k_loc(normed_X).view(B, L, self.num_kv_heads, self.head_dim)
        V = self.W_v_loc(normed_X).view(B, L, self.num_kv_heads, self.head_dim)

        Q = self.q_norm(Q)
        K = self.k_norm(K)

        # 🌟 基於物理隔離的模組，利用 FFT 頻域相位動態生成 RoPE 的相位偏置量
        delta_theta = fft_phase_lock_module(normed_X, layer_idx)
        
        # 進行相位鎖定之旋轉位置編碼
        Q = apply_phase_locked_rope(Q, self.head_dim, base_freq=self.rope_base, delta_theta=delta_theta).contiguous()
        
        # K 亦套用對應的平均相位偏移量
        delta_theta_k = delta_theta.mean(dim=2, keepdim=True) # shape: (B, L, 1, 128)
        K = apply_phase_locked_rope(K, self.head_dim, base_freq=self.rope_base, delta_theta=delta_theta_k).contiguous()
        
        V = V.contiguous()

        window_size = (-1, -1) if self.is_global else (self.C - 1, 0)

        attn_out = flash_attn_func(
            Q, K, V, dropout_p=0.0, causal=True, window_size=window_size
        )

        Z_hat = self.o_proj_loc(attn_out.reshape(B, L, self.num_heads * self.head_dim))

        X_res1 = X + self.post_attention_layernorm(Z_hat)
        normed_X_ffn = self.pre_feedforward_layernorm(X_res1)
        ffn_out = self.ffn(normed_X_ffn)
        Output = X_res1 + self.post_feedforward_layernorm(ffn_out)
        return Output
        
    return forward

# ==========================================
# [ 硬體保護：主動式散熱控制器 ]
# ==========================================
class ThermalControlCallback(TrainerCallback):
    def __init__(self, delay_seconds=1.5):
        self.delay_seconds = delay_seconds

    def on_step_end(self, args, state, control, **kwargs):
        time.sleep(self.delay_seconds)

# ==========================================
# [ 量子退火監控器 (FFT 專用無閘門簡化版) ]
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
                        if len(row) > 3 and row[3].strip(): 
                            try:
                                val = float(row[3])
                                if val < self.best_eval_loss: self.best_eval_loss = val
                            except ValueError: pass
                if self.best_eval_loss != float('inf'):
                    print(f"\n📈 [Monitor] 當前最佳 Eval Loss 基準: {self.best_eval_loss:.4f}")
            except Exception as e:
                print(f"\n⚠️ [Monitor] 讀取歷史紀錄失敗: {e}")
        else:
            with open(self.path, 'w', newline='') as f:
                csv.writer(f).writerow(['step', 'loss', 'ce_loss', 'eval_loss', 'time'])

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            model = kwargs.get('model', None)
            ce_loss_val = ""
            if model is not None:
                raw_model = model.module if hasattr(model, 'module') else model
                if hasattr(raw_model, '_last_ce_loss'):
                    ce_loss_val = f"{raw_model._last_ce_loss:.6f}"
                    logs["ce_loss"] = raw_model._last_ce_loss
            
            with open(self.path, 'a', newline='') as f:
                csv.writer(f).writerow([
                    state.global_step, 
                    logs.get("loss", ""), 
                    ce_loss_val,
                    logs.get("eval_loss", ""), 
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
                    best_model_path = os.path.join(self.save_dir, "best_cpt_model.pth")
                    raw_model = model.module if hasattr(model, 'module') else model
                    torch.save(raw_model.state_dict(), best_model_path)
                    print(f"\n[Monitor] 🌟 收斂突破 ({old_best:.4f} -> {current_eval_loss:.4f})，儲存至 {best_model_path}")
            else:
                print(f"\n[Monitor] 🛡️ 此次成績 ({current_eval_loss:.4f}) 未超越歷史最佳 ({self.best_eval_loss:.4f})。")

# ==========================================
# [ HuggingFace 自動上傳器 ]
# ==========================================
class HFAutoUploadCallback(TrainerCallback):
    def __init__(self, repo_id=REPO_ID, model_path=None, log_path=LOG_PATH,
                 ce_loss_threshold=HF_CE_LOSS_THRESHOLD, save_dir=SAVE_DIR):
        self.repo_id = repo_id
        self.model_path = model_path or os.path.join(save_dir, "best_cpt_model.pth")
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
# [ 物理隔離 ] N -> N+B 絕對斷層拼接器
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
    print("\n🚀 啟動 GEMMA3 FROM SCRATCH 訓練矩陣 (1D Conv + Sinc Causal 1K 版本，全開放訓練)...")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    parquet_file = os.path.join(DATASET_DIR, "agiv2_stage1_N_B.parquet")
    if not os.path.exists(parquet_file):
        raise FileNotFoundError(f"找不到實體矩陣 {parquet_file}，請先執行 cpt1K.py 產生資料。")
    
    print(f"📦 成功鎖定實體磁區，直接掛載 Parquet 拓撲: {parquet_file}")
    dataset_dict = load_dataset("parquet", data_files=parquet_file)
    
    dataset = dataset_dict['train'].train_test_split(test_size=50, seed=RANDOM_SEED)
    train_ds, eval_ds = dataset['train'], dataset['test']
    print(f"✅ 資料映射與絕對隔離完成！訓練集: {len(train_ds):,} 筆 | 獨立驗證集: {len(eval_ds)} 筆")

    # ==========================================
    # 🌟 核心替換：改用 GEMMA3，參數維持 CHUNK=256，隨機初始化
    # ==========================================
    base = GEMMA3(vocab_size=262144, D=1152, C=CHUNK, hidden_dim=6912, num_blocks=26, rope_local=ROPE_LOCAL, rope_global=ROPE_GLOBAL)
    print("\n🟢 已隨機初始化 GEMMA3 架構 (無預訓練權重)")
    
    # 實例化物理隔離的 Sinc Causal 調製模組
    print(f"\n🔮 正在初始化物理隔離之 Sinc Causal 調製模組 (總參數量精準等於 745,472)...")
    fft_phase_lock = SincCausalRoPE(num_layers=26, num_heads=4, head_dim=256, target_params=745472)
    
    # 動態修補 (Monkey Patch) 所有 Block 的 forward 以引入 FFT 相位鎖定調製
    print("🛠️ 正在動態修補模型解碼塊 (Gemma3DecoderLayer)，接合相位鎖定 RoPE 邏輯...")
    for idx, block in enumerate(base.blocks):
        block.forward = types.MethodType(
            make_phase_locked_decoder_forward(idx, fft_phase_lock),
            block
        )
        
    model = AGIV2GForCausalLMT(base, use_gc=True, gate_entropy_lambda=GATE_ENTROPY_LAMBDA)
    
    # 將實體隔離的調製器註冊為 model 的 submodule 以參與訓練、優化與存檔
    model.fft_phase_lock = fft_phase_lock
    print("✅ 模型實例化完成，物理隔離調製器已順利掛載。")
    
    # 🟢 FROM SCRATCH: 開放全局所有參數進行從頭訓練
    print("\n🟢 FROM SCRATCH: 開放全局所有參數進行從頭訓練...")
    for param in model.parameters():
        param.requires_grad = True
        
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"🟢 可訓練參數總量 (From Scratch): {trainable_params:,}")
            
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

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=CPTDataCollator(pad_id),   
        callbacks=[
            QuantumCPTMonitor(path=LOG_PATH, save_dir=SAVE_DIR),
            ThermalControlCallback(delay_seconds=0.0), # 🛡️ 散熱延遲設為 0
            HFAutoUploadCallback()                     # 🌐 自動上傳
        ]
    )

    last_checkpoint = get_last_checkpoint(SAVE_DIR)
    if last_checkpoint is not None:
        print(f"\n🚀 偵測到 1K 時間錨點 {last_checkpoint}，啟動狀態無損恢復...")
        trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        print("\n🚀 開始全新 1K From-Scratch 訓練...")
        trainer.train()
        
    torch.save(model.state_dict() if not hasattr(model, 'module') else model.module.state_dict(), os.path.join(SAVE_DIR, "final_agigemma3_scratch_1k_fftrope.pth"))

if __name__ == "__main__":
    main()
