# trainCPTAC_AGI_GEMMA3_1K_loraheaddual_scratch.py — AGI GEMMA 3 LoRA + 頭級子空間正交解耦 Hybrid From-Scratch 訓練引擎 (全開放訓練)
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
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    set_seed,
    TrainerCallback,
)
from transformers.trainer_utils import get_last_checkpoint
from datasets import load_dataset 

# 🌟 官方 FlashAttention 導入
try:
    from flash_attn import flash_attn_func
except ImportError:
    raise ImportError("請確定已安裝 flash-attn 套件: pip install flash-attn --no-build-isolation")

from GEMMA3 import GEMMA3 
from utils import AGIV2GForCausalLMT

RANDOM_SEED = 2026
set_seed(RANDOM_SEED)

# ==========================================
# [ 全局配置區 ] 
# ==========================================
MODEL_ID = "google/gemma-3-1b-it" 
DATASET_DIR = "~/agiv2_stage1_1K"            
SAVE_DIR = "~/agigemma3_scratch_checkpoints_1K_loraheaddual"
LOG_PATH = "~/agigemma3_scratch_1k_loraheaddual_log.csv"

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
REPO_ID = "tim9510019/AGIGEMMA3-1B-SCRATCH_1K_loraheaddual"
HF_CE_LOSS_THRESHOLD = 9.9          # 從頭訓練初期 Loss 較高，將門檻放寬

# ==========================================
# [ GQA 輔助函數 ]
# ==========================================
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    B, L, num_kv_heads, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, :, None, :].expand(B, L, num_kv_heads, n_rep, head_dim)
    return hidden_states.reshape(B, L, num_kv_heads * n_rep, head_dim)

# ==========================================
# [ LoRA 輕量化架構實現 (50% 參數: 372,736) ]
# ==========================================
class LoRALinear(nn.Module):
    def __init__(self, base_linear, r=4, alpha=8):
        super().__init__()
        self.base_linear = base_linear
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        
        in_features = base_linear.in_features
        out_features = base_linear.out_features
        
        # 設置 LoRA 可訓練參數 (採用 float32 儲存以維持精準梯度)
        self.lora_A = nn.Parameter(torch.empty(r, in_features, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.empty(out_features, r, dtype=torch.float32))
        
        # 參數初始化
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
    def forward(self, x):
        # 動態投射 LoRA 參數至輸入精度 (通常是 bfloat16)
        lora_A = self.lora_A.to(x.dtype)
        lora_B = self.lora_B.to(x.dtype)
        
        base_out = self.base_linear(x)
        lora_out = (F.linear(F.linear(x, lora_A), lora_B)) * self.scaling
        return base_out + lora_out

def apply_lora_to_model(base_model, r=4, alpha=8):
    """
    對 model 中的所有 26 個解碼塊 of W_q_loc 與 W_v_loc 進行 LoRA 注入。
    總參數增加量: 26 * 4 * (1152 + 1024 + 1152 + 256) = 372,736 個參數。
    """
    for i, block in enumerate(base_model.blocks):
        # 注入 W_q_loc (in: 1152, out: 1024)
        block.W_q_loc = LoRALinear(block.W_q_loc, r=r, alpha=alpha)
        # 注入 W_v_loc (in: 1152, out: 256)
        block.W_v_loc = LoRALinear(block.W_v_loc, r=r, alpha=alpha)
    return base_model

# ==========================================
# [ 物理隔離：時域 Sinc Causal 卷積模組 (1個 Wave 頭調製器, params: 372,736) ]
# ==========================================
class SincCausalRoPE(nn.Module):
    def __init__(self, num_layers=26, target_params=372736, W=64):
        """
        頭級解耦專用的波動頭調製器。
        總參數量: 372,736，每層配額 14,336。
        重塑為 112 x 128 (輸入 112 特徵通道，投射至 128 相位維度，對應 1 個 Wave Head)。
        """
        super().__init__()
        self.weight = nn.Parameter(torch.empty(target_params, dtype=torch.float32))
        nn.init.normal_(self.weight, mean=0.0, std=0.02)
        
        self.num_layers = num_layers
        self.params_per_layer = target_params // num_layers # 14,336
        self.W = W
        
    def forward(self, x, layer_idx):
        B, L, D = x.shape
        device = x.device
        dtype = x.dtype
        
        start_idx = layer_idx * self.params_per_layer
        end_idx = start_idx + self.params_per_layer
        w_layer = self.weight[start_idx:end_idx].view(112, 128)
        
        with torch.autocast(device_type="cuda", enabled=False):
            x_slice = x[..., :112].to(torch.float32)
            w_layer_cast = w_layer.to(torch.float32)
            
            h = torch.matmul(x_slice, w_layer_cast) # shape: (B, L, 128)
            
            t = torch.arange(self.W, device=device, dtype=torch.float32)
            t_0 = (self.W - 1) / 2.0
            
            fc = torch.linspace(0.02, 0.48, steps=128, device=device, dtype=torch.float32)
            
            args = 2.0 * fc.unsqueeze(1) * (t.unsqueeze(0) - t_0)
            sinc_val = torch.sinc(args) # [128, W]
            
            window = 0.54 - 0.46 * torch.cos(2.0 * math.pi * t / (self.W - 1))
            window = window.unsqueeze(0)
            
            kernel = 2.0 * fc.unsqueeze(1) * sinc_val * window # [128, W]
            kernel = kernel / (kernel.norm(p=1, dim=1, keepdim=True) + 1e-8)
            
            weight_conv = kernel.unsqueeze(1)
            
            h_t = h.transpose(1, 2)
            h_padded = F.pad(h_t, (self.W - 1, 0))
            
            h_modulated = F.conv1d(h_padded, weight_conv, groups=128)
            h_modulated = h_modulated.transpose(1, 2) # [B, L, 128]
            
            h_reshaped = h_modulated.view(B, L, 1, 128) # 對應 1 個頭的 (B, L, 1, 128) 旋轉角度偏置
            
        return h_reshaped.to(dtype)

# =======================================================
# [ 位置旋轉編碼實現 ]
# =======================================================
def apply_phase_locked_rope(x, head_dim, base_freq=1000000.0, delta_theta=None):
    B, L, num_heads, D = x.shape
    position = torch.arange(L, device=x.device).unsqueeze(1).float()
    div_term = torch.exp(
        torch.arange(0, head_dim, 2, device=x.device).float() * -(math.log(base_freq) / head_dim)
    )
    freqs = position * div_term # shape: (L, 128)
    emb = freqs.unsqueeze(0).unsqueeze(2) # shape: (1, L, 1, 128)
    
    if delta_theta is not None:
        emb = emb + delta_theta
        
    emb = torch.cat((emb, emb), dim=-1) # shape: (B, L, num_heads, 256)
    sin_val = torch.sin(emb).to(x.dtype)
    cos_val = torch.cos(emb).to(x.dtype)
    x1 = x[..., : head_dim // 2]
    x2 = x[..., head_dim // 2 :]
    x_rotated = torch.cat((-x2, x1), dim=-1)
    return (x * cos_val) + (x_rotated * sin_val)

def apply_fra_only(x, head_dim, delta_theta=None):
    B, L, num_heads, D = x.shape
    if delta_theta is not None:
        emb = delta_theta
    else:
        return x
    emb = torch.cat((emb, emb), dim=-1)
    sin_val = torch.sin(emb).to(x.dtype)
    cos_val = torch.cos(emb).to(x.dtype)
    x1 = x[..., : head_dim // 2]
    x2 = x[..., head_dim // 2 :]
    x_rotated = torch.cat((-x2, x1), dim=-1)
    return (x * cos_val) + (x_rotated * sin_val)

# ==========================================
# [ 動態修補前向傳播：頭級子空間正交解耦前向 ]
# ==========================================
def make_head_decoupled_decoder_forward(layer_idx, fft_phase_lock_module):
    def forward(self, X, shift_size=0, n_split_index=None):
        B, L, D = X.shape
        normed_X = self.input_layernorm(X)
        
        Q = self.W_q_loc(normed_X).view(B, L, self.num_heads, self.head_dim)
        K = self.W_k_loc(normed_X).view(B, L, self.num_kv_heads, self.head_dim)
        V = self.W_v_loc(normed_X).view(B, L, self.num_kv_heads, self.head_dim)
        
        Q = self.q_norm(Q)
        K = self.k_norm(K)
        
        # 1. 手動將 Key 和 Value 展開對齊 Query 的 4 個頭 (GQA 解耦)
        K = repeat_kv(K, self.num_key_value_groups) # shape: (B, L, 4, 256)
        V = repeat_kv(V, self.num_key_value_groups) # shape: (B, L, 4, 256)
        
        # 2. 切分 75% 粒子頭 (Heads 0,1,2) 與 25% 波動頭 (Head 3)
        Q_part, Q_wave = Q[:, :, :3], Q[:, :, 3:]
        K_part, K_wave = K[:, :, :3], K[:, :, 3:]
        
        # 3. 粒子頭：套用純粹標準的幾何空間 RoPE (無相位調製，確保剛性幾何防禦與 CE 收斂)
        Q_part = apply_phase_locked_rope(Q_part, self.head_dim, base_freq=self.rope_base, delta_theta=None)
        K_part = apply_phase_locked_rope(K_part, self.head_dim, base_freq=self.rope_base, delta_theta=None)
        
        # 4. 波動頭：套用 Pure FRA (無任何幾何絕對座標，徹底隔離 OOD 干擾)
        delta_theta = fft_phase_lock_module(normed_X, layer_idx) # shape: (B, L, 1, 128)
        Q_wave = apply_fra_only(Q_wave, self.head_dim, delta_theta=delta_theta)
        K_wave = apply_fra_only(K_wave, self.head_dim, delta_theta=delta_theta)
        
        # 5. 頭級特徵正交重組
        Q = torch.cat([Q_part, Q_wave], dim=2).contiguous()
        K = torch.cat([K_part, K_wave], dim=2).contiguous()
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
# [ 訓練回調與監控模組 ]
# ==========================================
class ThermalControlCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        pass # 🛡️ From-Scratch 散熱延遲設為 0

class QuantumCPTMonitor(TrainerCallback):
    def __init__(self, path=LOG_PATH, save_dir=SAVE_DIR):
        self.path = path
        self.save_dir = save_dir
        self.best_eval_loss = float('inf')
        os.makedirs(self.save_dir, exist_ok=True)
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    reader = csv.reader(f)
                    next(reader, None)
                    for row in reader:
                        if len(row) > 3 and row[3].strip():
                            val = float(row[3])
                            if val < self.best_eval_loss: self.best_eval_loss = val
            except: pass
        else:
            with open(self.path, "w", newline="") as f:
                csv.writer(f).writerow(["step", "loss", "ce_loss", "eval_loss", "time"])
                
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            model = kwargs.get("model", None)
            ce_loss_val = ""
            if model is not None:
                raw_model = model.module if hasattr(model, 'module') else model
                if hasattr(raw_model, "_last_ce_loss"):
                    ce_loss_val = f"{raw_model._last_ce_loss:.6f}"
            with open(self.path, "a", newline="") as f:
                csv.writer(f).writerow([state.global_step, logs.get("loss", ""), ce_loss_val, logs.get("eval_loss", ""), time.ctime()])
                
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics and "eval_loss" in metrics:
            current_eval_loss = metrics["eval_loss"]
            if current_eval_loss < self.best_eval_loss:
                self.best_eval_loss = current_eval_loss
                model = kwargs.get("model", None)
                if model is not None:
                    best_model_path = os.path.join(self.save_dir, "best_cpt_model.pth")
                    raw_model = model.module if hasattr(model, 'module') else model
                    torch.save(raw_model.state_dict(), best_model_path)

class HFAutoUploadCallback(TrainerCallback):
    def __init__(self, repo_id=REPO_ID, model_path=None, log_path=LOG_PATH, ce_loss_threshold=HF_CE_LOSS_THRESHOLD, save_dir=SAVE_DIR):
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
                if (step % 1000 == 0) or (step <= 100):
                    if os.path.isfile(self.model_path):
                        api.upload_file(path_or_fileobj=self.model_path, path_in_repo=os.path.basename(self.model_path), repo_id=self.repo_id, repo_type="model")
                if os.path.isfile(self.log_path):
                    api.upload_file(path_or_fileobj=self.log_path, path_in_repo=os.path.basename(self.log_path), repo_id=self.repo_id, repo_type="model")
            except: pass
                
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None: return
        ce_loss = metrics.get("eval_loss", None)
        if ce_loss is None: return
        if ce_loss < self.best_ce_loss and ce_loss < self.ce_loss_threshold:
            self.best_ce_loss = ce_loss
            t = threading.Thread(target=self._do_upload, args=(ce_loss, state.global_step), daemon=True)
            t.start()

# ==========================================
# [ 數據集與 Collator ]
# ==========================================
class CPTAC1KDataset(torch.utils.data.Dataset):
    def __init__(self, arrow_dataset):
        self.dataset = arrow_dataset
    def __len__(self): return len(self.dataset)
    def __getitem__(self, idx):
        item = self.dataset[idx]
        input_ids = torch.tensor(item["input_ids"], dtype=torch.long)
        return {"input_ids": input_ids, "labels": input_ids.clone()}

class CPTDataCollator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id
    def __call__(self, features):
        batch_input_ids, batch_labels, batch_n_splits = [], [], []
        max_len = max(len(f["input_ids"]) for f in features)
        for f in features:
            seq = f["input_ids"]
            n_split = min(f["n_split_index"], len(seq)) if "n_split_index" in f else len(seq)
            pad_len = max_len - len(seq)
            batch_input_ids.append(seq + [self.pad_token_id] * pad_len)
            batch_labels.append([-100] * n_split + seq[n_split:] + [-100] * pad_len)
            batch_n_splits.append(n_split)
        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
            "n_split_index": torch.tensor(batch_n_splits, dtype=torch.long),
        }

def main():
    print(f"🔮 初始化 GEMMA 3 1B LoRA + HEAD_DUAL Hybrid From-Scratch 訓練引擎...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    parquet_file = os.path.join(DATASET_DIR, "agiv2_stage1_N_B.parquet")
    dataset_dict = load_dataset("parquet", data_files=parquet_file)
    dataset = dataset_dict["train"].train_test_split(test_size=50, seed=RANDOM_SEED)
    train_ds, eval_ds = dataset["train"], dataset["test"]

    base = GEMMA3(vocab_size=262144, D=1152, C=CHUNK, hidden_dim=6912, num_blocks=26, rope_local=ROPE_LOCAL, rope_global=ROPE_GLOBAL)
    
    # 1. 注入 50% 參數量的 LoRA 模組 (rank=4, alpha=8, params=372,736)
    print("\n💉 正在注入 LoRA 微調模組 (r=4, alpha=8) 至 W_q_loc 與 W_v_loc...")
    base = apply_lora_to_model(base, r=4, alpha=8)
    
    # 2. 注入 50% 參數量的頭級子空間正交解耦調製器 (params: 372,736, weight shape: 112 x 128)
    print(f"\n🔮 正在初始化物理隔離之 Sinc Causal 調製模組 (總參數量精準等於 372,736)...")
    fft = SincCausalRoPE(num_layers=26, target_params=372736)
    
    for idx, block in enumerate(base.blocks):
        block.forward = types.MethodType(make_head_decoupled_decoder_forward(idx, fft), block)
        
    model = AGIV2GForCausalLMT(base, use_gc=True, gate_entropy_lambda=GATE_ENTROPY_LAMBDA)
    model.fft_phase_lock = fft
    print("✅ 模型實例化完成，LoRA 與頭級正交解耦 Duality 調製器已順利掛載。")
    
    # FROM SCRATCH: 開放全局所有參數進行從頭訓練
    for param in model.parameters():
        param.requires_grad = True
            
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"🟢 可訓練參數總量 (From Scratch Hybrid): {trainable_params:,}")
    
    if trainable_params != 1302621312:
        raise ValueError(f"❌ 參數量偏差！實際 {trainable_params:,}")
    print(f"✅ 參數量拓撲完全契合！")
            
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
        report_to="none",
    )
    
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=CPTDataCollator(pad_id),
        callbacks=[
            QuantumCPTMonitor(path=LOG_PATH, save_dir=SAVE_DIR),
            ThermalControlCallback(),
            HFAutoUploadCallback()
        ]
    )
    
    last_checkpoint = get_last_checkpoint(SAVE_DIR)
    if last_checkpoint is not None:
        trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        trainer.train()

    torch.save(
        model.state_dict() if not hasattr(model, "module") else model.module.state_dict(),
        os.path.join(SAVE_DIR, "final_agigemma3_scratch_1k_loraheaddual.pth"),
    )

if __name__ == "__main__":
    main()
