# trainCPTAC_AGI_GEMMA3_1K_dnagatedual_scratch.py — AGI GEMMA 3 DNA 仿生雙螺旋糾纏門控 DNA_GateDual From-Scratch 訓練引擎
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
SAVE_DIR = "~/agigemma3_scratch_checkpoints_1K_dnagatedual_ablation"
LOG_PATH = "~/agigemma3_scratch_1k_dnagatedual_ablation_log.csv"

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
GATE_ENTROPY_LAMBDA = 0.0          
ROPE_LOCAL = 10000.0
ROPE_GLOBAL = 1000000.0

# HuggingFace 自動上傳配置
REPO_ID = "tim9510019/AGIGEMMA3-1B-SCRATCH_1K_dnagatedual_ablation"
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
# [ DNAGatedSincCausalRoPE: 745,472 參數對齊 ]
# ==========================================
class DNAGatedSincCausalRoPE(nn.Module):
    def __init__(self, num_layers=26, target_params=745472, W=64):
        """
        DNA_GateDual 專用雙螺旋鹼基對門控調製模組。
        參數嚴格鎖死 745,472。在每層中進行精確的仿生對齊：
        1. 波動頭 Sinc 相位投影權重: 152 x 128 = 19,456 參數
        2. 鹼基對雙向門控投影權重: 1152 x 8 = 9,216 參數
        每層共 28,672，26 層總計 745,472！
        """
        super().__init__()
        sinc_total = 152 * 128 * num_layers # 505,856
        gate_total = 1152 * 8 * num_layers # 239,616
        
        self.weight = nn.Parameter(torch.empty(sinc_total, dtype=torch.float32))
        self.gate_weight = nn.Parameter(torch.empty(gate_total, dtype=torch.float32))
        
        nn.init.normal_(self.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.gate_weight, mean=0.0, std=0.02)
        
        self.num_layers = num_layers
        self.sinc_per_layer = 152 * 128 # 19,456
        self.gate_per_layer = 1152 * 8 # 9,216
        self.W = W
        
    def forward(self, x, layer_idx):
        B, L, D = x.shape
        device = x.device
        
        # 1. 獲取波動 Sinc 卷積投射權重 (152 x 128)
        sinc_start = layer_idx * self.sinc_per_layer
        sinc_end = sinc_start + self.sinc_per_layer
        w_layer = self.weight[sinc_start:sinc_end].view(152, 128)
        
        # 2. 獲取動態鹼基對閘門投影權重 (1152 x 8)
        gate_start = layer_idx * self.gate_per_layer
        gate_end = gate_start + self.gate_per_layer
        w_gate = self.gate_weight[gate_start:gate_end].view(1152, 8)
        
        with torch.autocast(device_type="cuda", enabled=False):
            # Q-slice 寬度縮至 152 進行波動特徵變換
            x_slice = x[..., :152].to(torch.float32)
            w_layer_cast = w_layer.to(torch.float32)
            
            h = torch.matmul(x_slice, w_layer_cast) # shape: (B, L, 128)
            
            # Sinc 時域卷積調製
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
            
            delta_theta = h_modulated.view(B, L, 1, 128) # 對應 1 個頭的旋轉角度偏置 (B, L, 1, 128)
            
            # 計算 8 維鹼基對動態門控向量 gate
            x_cast = x.to(torch.float32)
            w_gate_cast = w_gate.to(torch.float32)
            
            s = torch.matmul(x_cast, w_gate_cast) # shape: (B, L, 8)
            gate = torch.sigmoid(s) # shape: (B, L, 8)
            
        return delta_theta.to(x.dtype), gate.to(x.dtype)

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
# [ 同層雙向鹼基對糾纏前向補丁 ]
# ==========================================
def make_dna_gated_decoupled_decoder_forward(layer_idx, gated_module):
    def forward(self, X, shift_size=0, n_split_index=None):
        B, L, D = X.shape
        normed_X = self.input_layernorm(X)
        
        Q = self.W_q_loc(normed_X).view(B, L, self.num_heads, self.head_dim)
        K = self.W_k_loc(normed_X).view(B, L, self.num_kv_heads, self.head_dim)
        V = self.W_v_loc(normed_X).view(B, L, self.num_kv_heads, self.head_dim)
        
        Q = self.q_norm(Q)
        K = self.k_norm(K)
        
        # 1. Key 與 Value 展開對齊 Query 4 個頭
        K = repeat_kv(K, self.num_key_value_groups) # shape: (B, L, 4, 256)
        V = repeat_kv(V, self.num_key_value_groups) # shape: (B, L, 4, 256)
        
        # 2. 75% 幾何粒子頭 (0,1,2) 與 25% 波動頭 (3)
        Q_part, Q_wave = Q[:, :, :3], Q[:, :, 3:]
        K_part, K_wave = K[:, :, :3], K[:, :, 3:]
        
        # 3. 粒子頭：標準幾何空間 RoPE (無相位調製)
        Q_part = apply_phase_locked_rope(Q_part, self.head_dim, base_freq=self.rope_base, delta_theta=None)
        K_part = apply_phase_locked_rope(K_part, self.head_dim, base_freq=self.rope_base, delta_theta=None)
        
        # 4. 波動頭：套用 Pure FRA 與動態自適應鹼基對閘門計算
        delta_theta, gate = gated_module(normed_X, layer_idx) # delta_theta: (B, L, 1, 128), gate: (B, L, 8)
        Q_wave = apply_fra_only(Q_wave, self.head_dim, delta_theta=delta_theta)
        K_wave = apply_fra_only(K_wave, self.head_dim, delta_theta=delta_theta)
        
        # 5. 標準注意力計算
        Q = torch.cat([Q_part, Q_wave], dim=2).contiguous()
        K = torch.cat([K_part, K_wave], dim=2).contiguous()
        V = V.contiguous()
        
        window_size = (-1, -1) if self.is_global else (self.C - 1, 0)
        
        attn_out = flash_attn_func(
            Q, K, V, dropout_p=0.0, causal=True, window_size=window_size
        ) # shape: (B, L, 4, 256)
        
        attn_part = attn_out[:, :, :3] # shape: (B, L, 3, 256)
        attn_wave = attn_out[:, :, 3:] # shape: (B, L, 1, 256)
        
        # =============================================================
        # 🧬 6. 同層雙螺旋正交子空間路由 (Complementary Subspace Gated Routing - v2)
        # =============================================================
        # 將 Q, K, V 的 256 特徵維度均分為 8 個子空間組（每組 32 維）
        attn_part_grouped = attn_part.reshape(B, L, 3, 8, 32)
        attn_wave_grouped = attn_wave.reshape(B, L, 1, 8, 32)
        
        # 粒子門控 g_P = 1.0 - gate, 互補波動門控 g_W = gate
        gate_w = gate.unsqueeze(2).unsqueeze(-1) # shape: (B, L, 1, 8, 1)
        gate_p = 1.0 - gate_w
        
        # 🧬 雙鏈第一性原理互補正交路由：兩鏈各佔互補子空間，不進行跨鏈加性混合以防表徵抹平！
        attn_part_gated = (attn_part_grouped * gate_p).reshape(B, L, 3, 256)
        attn_wave_gated = (attn_wave_grouped * gate_w).reshape(B, L, 1, 256)
        
        # 重組正交特徵流
        attn_out_gated = torch.cat([attn_part_gated, attn_wave_gated], dim=2)
        
        # 7. 重組投影
        Z_hat = self.o_proj_loc(attn_out_gated.reshape(B, L, self.num_heads * self.head_dim))
        
        X_res1 = X + self.post_attention_layernorm(Z_hat)
        normed_X_ffn = self.pre_feedforward_layernorm(X_res1)
        ffn_out = self.ffn(normed_X_ffn)
        Output = X_res1 + self.post_feedforward_layernorm(ffn_out)
        
        # 🧬 8. 計算本層的閘門負熵 (無 side-channel 洩漏)
        with torch.autocast(device_type="cuda", enabled=False):
            eps = 1e-8
            gate_32 = gate.to(torch.float32)
            p = torch.stack([gate_32, 1.0 - gate_32], dim=0) # shape: (2, B, L, 8)
            p = p / (p.sum(dim=0, keepdim=True) + eps)
            block_entropy = (p * torch.log(p + eps)).sum(dim=0).mean() # scalar
            
        return Output, block_entropy.to(X.dtype)
    return forward

# ==========================================
# [ 完美無記憶體洩漏的 CausalLM 包裝器 ]
# ==========================================
class DNAGatedAGIV2GForCausalLMT(nn.Module):
    def __init__(self, base_model, use_gc=True, gate_entropy_lambda=0.01):
        super().__init__()
        self.base_model = base_model
        self.use_gc = use_gc
        self.gate_entropy_lambda = gate_entropy_lambda
        self._keys_to_ignore_on_save = None
        self._last_ce_loss = 0.0
        self._last_gate_entropy_loss = 0.0

    def forward(self, input_ids, labels=None, **kwargs):
        n_split_index = kwargs.get("n_split_index", None)

        hidden_states = self.base_model.embedding(input_ids)
        hidden_states = hidden_states * math.sqrt(self.base_model.D)

        gate_entropies = []
        import torch.utils.checkpoint as checkpoint
        
        for i, block in enumerate(self.base_model.blocks):
            shift_size = (block.C // 2) if (i % 2 != 0) else 0
            if self.training and self.use_gc:
                hidden_states, block_entropy = checkpoint.checkpoint(
                    block,
                    hidden_states,
                    shift_size,
                    use_reentrant=False,
                    n_split_index=n_split_index,
                )
            else:
                hidden_states, block_entropy = block(
                    hidden_states, shift_size=shift_size, n_split_index=n_split_index
                )
            gate_entropies.append(block_entropy)

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

                c_loss = loss_fct(
                    c_logits.reshape(-1, c_logits.size(-1)).float(),
                    c_labels.reshape(-1),
                )
                total_loss += c_loss

                valid_tokens += (c_labels != -100).sum().item()
                del c_logits, c_hidden

            loss = total_loss / max(valid_tokens, 1)

            if self.training:
                # 🌟 第一性原理自適應負熵正則，完全排除 memory leak 旁路
                if len(gate_entropies) > 0:
                    gate_neg_entropy = torch.stack(gate_entropies).mean()
                    gate_entropy_loss = gate_neg_entropy + math.log(3.0)

                    self._last_ce_loss = loss.detach().item()
                    self._last_gate_entropy_loss = gate_entropy_loss.detach().item()

                    # 總 loss = CE loss + λ * 負熵懲罰
                    loss = loss + self.gate_entropy_lambda * gate_entropy_loss
                else:
                    self._last_ce_loss = loss.detach().item()
                    self._last_gate_entropy_loss = 0.0
            else:
                self._last_ce_loss = loss.detach().item()
                self._last_gate_entropy_loss = 0.0
        else:
            logits = self.base_model.fc_out(hidden_states)

        return (
            {"loss": loss, "logits": logits} if logits is not None else {"loss": loss}
        )

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
    print(f"🔮 初始化 GEMMA 3 1B DNA_GateDual (雙螺旋糾纏門控) From-Scratch 訓練引擎...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    parquet_file = os.path.join(DATASET_DIR, "agiv2_stage1_N_B.parquet")
    dataset_dict = load_dataset("parquet", data_files=parquet_file)
    dataset = dataset_dict["train"].train_test_split(test_size=50, seed=RANDOM_SEED)
    train_ds, eval_ds = dataset["train"], dataset["test"]

    base = GEMMA3(vocab_size=262144, D=1152, C=CHUNK, hidden_dim=6912, num_blocks=26, rope_local=ROPE_LOCAL, rope_global=ROPE_GLOBAL)
    
    # 實例化 DNA_GateDual 鹼基對雙向糾纏門控路由器 (params: 745,472)
    gated_module = DNAGatedSincCausalRoPE(num_layers=26, target_params=745472)
    base.gated_module_list = nn.ModuleList([gated_module])
    
    for idx, block in enumerate(base.blocks):
        block.forward = types.MethodType(make_dna_gated_decoupled_decoder_forward(idx, gated_module), block)
        
    # 🚀 動態修補 base.forward 以適應 tuple 輸出
    def new_base_forward(self, x, n_split_index=None):
        out = self.embedding(x)
        out = out * math.sqrt(self.D)
        for i, block in enumerate(self.blocks):
            out, _ = block(out, shift_size=0, n_split_index=n_split_index)
        out = self.final_norm(out)
        logits = self.fc_out(out)
        logits = logits / 30.0
        logits = torch.tanh(logits) * 30.0
        return logits

    base.forward = types.MethodType(new_base_forward, base)

    # 🚀 使用完美的無 side-channel 洩漏 CausalLM 封裝器
    model = DNAGatedAGIV2GForCausalLMT(base, use_gc=True, gate_entropy_lambda=GATE_ENTROPY_LAMBDA)
    # model.fft_phase_lock = gated_module (已移除以防止 Safetensors 記憶體共用錯誤)
    
    # FROM SCRATCH: 開放全局所有參數進行從頭訓練
    for param in model.parameters():
        param.requires_grad = True
            
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"🟢 可訓練參數總量 (From Scratch DNA_GateDual): {trainable_params:,}")
    
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
        os.path.join(SAVE_DIR, "final_agigemma3_scratch_1k_dnagatedual.pth"),
    )

if __name__ == "__main__":
    main()
