"""
trainDNA16B_distill.py — DNA Helix Ultimate 16B 蒸餾訓練主程序
================================================================
總計 50B Tokens，分三階段訓練：

  Phase 1 ── 1B  ── 純蒸餾暖身
    ・只餵 Data-Type A（≤1024 tok）
    ・波動頭與粒子頭在糾纏架構下協同蒸餾，共同學習 Qwen3-Coder Soft Labels
    ・WPCE 糾纏引擎引導雙鏈交互共振
    ・Loss：CE + KL（α=1.0）+ 負熵平衡門控分配（β=0.5）

  Phase 2 ── 1B  ── 純長文波動共振
    ・只餵 Data-Type B（32K~128K tok）
    ・波動頭 Sinc 卷積在長文中獨立建立相位共振模式
    ・粒子頭 RoPE 在長距自然退化，讓波動頭梯度乾淨主導
    ・Loss：CE + 強化負熵正則（β=5.0），不做 KL（α=0.0）

  Phase 3 ── 48B ── 雙螺旋混合共生
    ・A:B = 4:1 混合 Batch（每批 80% Type A + 20% Type B）
    ・WPCE 因果糾纏正式啟動：粒子頭引導 Δθ，波動頭 AW 反向修正
    ・Loss：A型 KL（α=1.0）+ B型 負熵（β=5.0）動態切換

核心特性：
  ・Sparse Top-128 KL Divergence（Teacher Logit Shard LRU 懶加載）
  ・思考段落 <think> 額外加權 KL × 1.5
  ・梯度 Checkpoint + 8-bit paged AdamW + bfloat16
  ・Phase 切換後自動 resume，無需重啟
"""

import os
import sys
import csv
import time
import math
import types
import json
import threading
import random
from pathlib import Path
from glob import glob

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

os.environ["BITSANDBYTES_NOWELCOME"]   = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"]  = "expandable_segments:True"

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file as st_load
from huggingface_hub import HfApi
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    set_seed,
    TrainerCallback,
)
from transformers.trainer_utils import get_last_checkpoint
from datasets import load_dataset, concatenate_datasets

RANDOM_SEED = 2026
set_seed(RANDOM_SEED)

# ============================================================
# 全局配置
# ============================================================

# ── 路徑配置 ──
DATASET_DIR_A   = os.path.expanduser("~/task06_data/type_A/")      # Data-Type A Parquet
DATASET_DIR_B   = os.path.expanduser("~/task06_data/type_B/")      # Data-Type B Parquet
LOGIT_SHARD_DIR = os.path.expanduser("~/task06_data/logit_shards/")# Teacher Logit Shards
SAVE_DIR        = os.path.expanduser("~/task06_checkpoints/")
LOG_PATH        = os.path.expanduser("~/task06_distill_log.csv")
TOKENIZER_ID    = "Qwen/Qwen3-Coder-30B-A3B-Instruct"

# ── HuggingFace 上傳 ──
REPO_ID              = "tim9510019/DNA-Helix-16B-Coder"
HF_CE_LOSS_THRESHOLD = 3.0

# ============================================================
# 三階段訓練設定（50B 總量）
# ============================================================
PHASE_CONFIGS = {
    1: {
        "name":                 "純蒸餾暖身",
        "target_tokens":        1_000_000_000,
        "dataset_mode":         "A_only",
        "learning_rate":        1e-4,
        "warmup_steps":         500,
        "alpha_distill":        1.0,
        "beta_wave_long":       0.5,
        "beta_wave_short":      0.5,
        "gate_entropy_lambda":  0.01,
        "batch_mix_ratio_a":    1.0,
        "chunk":                1024,
        "long_chunk":           1024,
        "eval_steps":           10000,
        "save_steps":           10000,
        "hf_threshold":         4.0,
    },
    2: {
        "name":                 "純長文波動共振",
        "target_tokens":        1_000_000_000,
        "dataset_mode":         "B_only",
        "learning_rate":        5e-5,
        "warmup_steps":         200,
        "alpha_distill":        0.0,
        "beta_wave_long":       5.0,
        "beta_wave_short":      5.0,
        "gate_entropy_lambda":  0.05,
        "batch_mix_ratio_a":    0.0,
        "chunk":                1024,
        "long_chunk":           32768,
        "eval_steps":           100,
        "save_steps":           100,
        "hf_threshold":         3.5,
    },
    3: {
        "name":                 "雙螺旋混合共生",
        "target_tokens":        48_000_000_000,
        "dataset_mode":         "mixed",
        "learning_rate":        3e-5,
        "warmup_steps":         1000,
        "alpha_distill":        1.0,
        "beta_wave_long":       5.0,
        "beta_wave_short":      1.0,
        "gate_entropy_lambda":  0.01,
        "batch_mix_ratio_a":    0.80,
        "chunk":                1024,
        "long_chunk":           32768,
        "eval_steps":           500,
        "save_steps":           500,
        "hf_threshold":         3.0,
    },
}

# ── 運行時從命令行指定當前 Phase ──
import argparse as _argparse
_parser = _argparse.ArgumentParser(add_help=False)
_parser.add_argument("--phase", type=int, default=1, choices=[1, 2, 3])
_args, _ = _parser.parse_known_args()
PHASE = _args.phase

# ── 根據 PHASE 載入對應超參數 ──
_cfg                 = PHASE_CONFIGS[PHASE]
DATASET_MODE         = _cfg["dataset_mode"]
TARGET_TOKENS        = _cfg["target_tokens"]
LEARNING_RATE        = _cfg["learning_rate"]
WARMUP_STEPS         = _cfg["warmup_steps"]
ALPHA_DISTILL        = _cfg["alpha_distill"]
BETA_WAVE_LONG       = _cfg["beta_wave_long"]
BETA_WAVE_SHORT      = _cfg["beta_wave_short"]
GATE_ENTROPY_LAMBDA  = _cfg["gate_entropy_lambda"]
BATCH_MIX_RATIO_A    = _cfg["batch_mix_ratio_a"]
CHUNK                = _cfg["chunk"]
LONG_CHUNK           = _cfg["long_chunk"]
EVAL_STEPS           = _cfg["eval_steps"]
SAVE_STEPS           = _cfg["save_steps"]
HF_CE_LOSS_THRESHOLD = _cfg["hf_threshold"]

# ── 固定超參數 ──
SAVE_TOTAL_LIMIT      = 2
LOGGING_STEPS         = 1
BATCH_SIZE_PER_DEVICE = 1
GRAD_ACCUMULATION_STEPS = 8
CHUNK_SIZE_EVAL       = 1024
DISTILL_TEMPERATURE   = 1.5
THINK_KL_WEIGHT       = 1.5
TOP_K_LOGITS          = 128

_eff_batch = BATCH_SIZE_PER_DEVICE * GRAD_ACCUMULATION_STEPS * max(CHUNK, 1)
MAX_STEPS = max(1, TARGET_TOKENS // _eff_batch)

print(f"\n{'='*60}")
print(f"🧬 DNA Helix 16B Coder — Phase {PHASE}：{_cfg['name']}")
print(f"   目標 Token 量 : {TARGET_TOKENS/1e9:.1f}B")
print(f"   預計訓練步數  : {MAX_STEPS:,}")
print(f"   Dataset Mode  : {DATASET_MODE}")
print(f"   LR={LEARNING_RATE} | α={ALPHA_DISTILL} | β_long={BETA_WAVE_LONG} | β_short={BETA_WAVE_SHORT}")
print(f"{'='*60}\n")

# ============================================================
# Sparse KL Divergence
# ============================================================

def sparse_kl_divergence(
    student_logits: torch.Tensor,
    teacher_values: torch.Tensor,
    teacher_indices: torch.Tensor,
    temperature: float = DISTILL_TEMPERATURE,
    think_mask: torch.Tensor = None,
    think_weight: float = THINK_KL_WEIGHT,
) -> torch.Tensor:
    B, L, vocab_size = student_logits.shape
    top_k = teacher_values.shape[-1]

    t_values  = teacher_values.to(torch.float32)
    t_indices = teacher_indices.to(torch.long)

    s_at_teacher = student_logits.float().gather(-1, t_indices)

    s_probs = F.softmax(s_at_teacher / temperature, dim=-1)
    t_probs = F.softmax(t_values / temperature, dim=-1)

    kl_per_token = F.kl_div(
        s_probs.log() + 1e-8,
        t_probs,
        reduction='none',
    ).sum(dim=-1)

    if think_mask is not None:
        weight = torch.ones_like(kl_per_token)
        weight[think_mask] = think_weight
        kl_per_token = kl_per_token * weight

    kl_loss = kl_per_token.mean()
    return kl_loss * (temperature ** 2)


# ============================================================
# Logit Shard 懶加載快取 (已修復 DDP Race Condition)
# ============================================================

class LogitShardCache:
    """
    懶加載 + LRU 快取 Teacher Logit Shards。
    一次最多快取 N_CACHE 份 Shard，並加入 Thread Lock 防止多卡 DDP 競賽條件。
    """
    def __init__(self, shard_dir: str, n_cache: int = 3):
        self.shard_dir = Path(shard_dir)
        self.n_cache   = n_cache
        self._cache    = {}       # {shard_id: {"logit_values": ..., "logit_indices": ...}}
        self._access   = {}       # {shard_id: timestamp}
        self._lock     = threading.Lock()  # 執行緒鎖

    def get(self, shard_id: int, offset: int, seq_len: int):
        if shard_id < 0:
            return None, None

        with self._lock:
            if shard_id not in self._cache:
                self._load_shard(shard_id)

            if shard_id not in self._cache:
                return None, None

            self._access[shard_id] = time.time()
            shard = self._cache[shard_id]

            try:
                values  = shard["logit_values"][offset, :seq_len, :]
                indices = shard["logit_indices"][offset, :seq_len, :]
                return values, indices
            except IndexError:
                return None, None

    def _load_shard(self, shard_id: int):
        if len(self._cache) >= self.n_cache:
            oldest_id = min(self._access, key=self._access.get)
            # 雙重確認防護
            if oldest_id in self._cache:
                del self._cache[oldest_id]
            if oldest_id in self._access:
                del self._access[oldest_id]

        shard_path = self.shard_dir / f"logit_shard_{shard_id:03d}.safetensors"
        if not shard_path.exists():
            return

        try:
            data = st_load(str(shard_path), device="cpu")
            self._cache[shard_id]  = data
            self._access[shard_id] = time.time()
        except Exception as e:
            print(f"⚠️  Logit Shard {shard_id} 載入失敗：{e}")


# ============================================================
# 雙流混合 Dataset
# ============================================================

class DualStreamDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        ds_A,
        ds_B,
        mode: str = DATASET_MODE,
        ratio_A: float = BATCH_MIX_RATIO_A,
        seed: int = RANDOM_SEED,
    ):
        self.ds_A = ds_A
        self.ds_B = ds_B
        self.mode = mode
        self.rng  = random.Random(seed)
        self._index_map = []

        n_A = len(ds_A) if ds_A is not None else 0
        n_B = len(ds_B) if ds_B is not None else 0

        if mode == "A_only":
            self._index_map = [("A", i) for i in range(n_A)]
            print(f"  [Phase 1 A_only] {n_A:,} 樣本 — 純蒸餾暖身，波粒糾纏協同蒸餾")
        elif mode == "B_only":
            self._index_map = [("B", i) for i in range(n_B)]
            print(f"  [Phase 2 B_only] {n_B:,} 樣本 — 純長文波動共振，beta={BETA_WAVE_LONG}")
        else:
            if ratio_A >= 1.0 or n_B == 0:
                self._index_map = [("A", i) for i in range(n_A)]
            else:
                ab_ratio = ratio_A / (1.0 - ratio_A)
                ia, ib, a_credit = 0, 0, 0.0
                while ia < n_A or ib < n_B:
                    a_credit += ab_ratio
                    while a_credit >= 1.0 and ia < n_A:
                        self._index_map.append(("A", ia))
                        ia += 1
                        a_credit -= 1.0
                    if ib < n_B:
                        self._index_map.append(("B", ib % n_B))
                        ib += 1
            print(f"  [Phase 3 mixed] {n_A:,} A + {n_B:,} B = {len(self._index_map):,} 樣本 "
                  f"(A:{ratio_A:.0%} B:{1-ratio_A:.0%}) — WPCE 雙螺旋共生")

    def __len__(self):
        return len(self._index_map)

    def __getitem__(self, idx):
        dtype, real_idx = self._index_map[idx]
        if dtype == "A":
            item = self.ds_A[real_idx]
            ids  = item["input_ids"]
            return {
                "input_ids":      torch.tensor(ids, dtype=torch.long),
                "labels":         torch.tensor(item["labels"], dtype=torch.long),
                "n_split_index":  item.get("n_split_index", len(ids)),
                "data_type":      "A",
                "logit_shard_id": item.get("logit_shard_id", -1),
                "logit_offset":   item.get("logit_offset",   -1),
                "think_mask":     torch.tensor(
                    item.get("think_mask", [False] * len(ids)), dtype=torch.bool),
            }
        else:
            item = self.ds_B[real_idx]
            ids  = item["input_ids"][:LONG_CHUNK]
            return {
                "input_ids":      torch.tensor(ids, dtype=torch.long),
                "labels":         torch.tensor(ids, dtype=torch.long),
                "n_split_index":  0,
                "data_type":      "B",
                "logit_shard_id": -1,
                "logit_offset":   -1,
                "think_mask":     torch.zeros(len(ids), dtype=torch.bool),
            }


# ============================================================
# Collator
# ============================================================

class DualStreamCollator:
    def __init__(self, pad_token_id: int):
        self.pad_id = pad_token_id

    def __call__(self, features):
        a_features = [f for f in features if f["data_type"] == "A"]
        b_features = [f for f in features if f["data_type"] == "B"]

        a_batch = self._pad_batch(a_features) if a_features else None
        b_batch = self._pad_batch(b_features) if b_features else None

        if a_batch and b_batch:
            batch = {
                "input_ids":      torch.cat([a_batch["input_ids"],   b_batch["input_ids"]], dim=0),
                "labels":         torch.cat([a_batch["labels"],      b_batch["labels"]], dim=0),
                "n_split_index":  torch.cat([a_batch["n_split_index"], b_batch["n_split_index"]]),
                "data_type_mask": torch.cat([
                    torch.zeros(len(a_features), dtype=torch.bool),
                    torch.ones(len(b_features), dtype=torch.bool),
                ]),
                "logit_shard_id": torch.cat([a_batch["logit_shard_id"], b_batch["logit_shard_id"]]),
                "logit_offset":   torch.cat([a_batch["logit_offset"],   b_batch["logit_offset"]]),
                "think_mask":     torch.cat([a_batch["think_mask"],     b_batch["think_mask"]], dim=0),
            }
        elif a_batch:
            batch = {**a_batch, "data_type_mask": torch.zeros(len(a_features), dtype=torch.bool)}
        else:
            batch = {**b_batch, "data_type_mask": torch.ones(len(b_features), dtype=torch.bool)}

        return batch

    def _pad_batch(self, features):
        max_len = max(len(f["input_ids"]) for f in features)
        input_ids_list, labels_list, n_splits = [], [], []
        shard_ids, offsets, think_masks = [], [], []

        for f in features:
            ids   = f["input_ids"]
            lbls  = f["labels"]
            mask  = f["think_mask"]
            pad   = max_len - len(ids)
            input_ids_list.append(torch.cat([ids, torch.full((pad,), self.pad_id, dtype=torch.long)]))
            labels_list.append(torch.cat([lbls, torch.full((pad,), -100, dtype=torch.long)]))
            think_masks.append(torch.cat([mask, torch.zeros(pad, dtype=torch.bool)]))
            n_splits.append(f["n_split_index"])
            shard_ids.append(f["logit_shard_id"])
            offsets.append(f["logit_offset"])

        return {
            "input_ids":      torch.stack(input_ids_list),
            "labels":         torch.stack(labels_list),
            "n_split_index":  torch.tensor(n_splits, dtype=torch.long),
            "logit_shard_id": torch.tensor(shard_ids, dtype=torch.long),
            "logit_offset":   torch.tensor(offsets, dtype=torch.long),
            "think_mask":     torch.stack(think_masks),
        }


# ============================================================
# DNA Helix 蒸餾損失 CausalLM Wrapper
# ============================================================

class DNAHelixDistillLM(nn.Module):
    def __init__(
        self,
        base_model,
        logit_shard_cache: LogitShardCache,
        use_gc: bool = True,
        gate_entropy_lambda: float = GATE_ENTROPY_LAMBDA,
        distill_temperature: float = DISTILL_TEMPERATURE,
        alpha_distill: float = ALPHA_DISTILL,
        beta_wave_long: float = BETA_WAVE_LONG,
        beta_wave_short: float = BETA_WAVE_SHORT,
    ):
        super().__init__()
        self.base_model          = base_model
        self.logit_cache         = logit_shard_cache
        self.use_gc              = use_gc
        self.gate_entropy_lambda = gate_entropy_lambda
        self.distill_temperature = distill_temperature
        self.alpha_distill       = alpha_distill
        self.beta_wave_long      = beta_wave_long
        self.beta_wave_short     = beta_wave_short

    def forward(self, input_ids, labels=None, **kwargs):
        data_type_mask = kwargs.get("data_type_mask", None)
        logit_shard_id = kwargs.get("logit_shard_id", None)
        logit_offset   = kwargs.get("logit_offset", None)
        think_mask     = kwargs.get("think_mask", None)
        n_split_index  = kwargs.get("n_split_index", None)

        import torch.utils.checkpoint as cp

        hidden_states = self.base_model.embedding(input_ids)
        hidden_states = hidden_states * math.sqrt(self.base_model.D)

        gate_entropies = []
        for i, block in enumerate(self.base_model.blocks):
            shift_size = (block.C // 2) if (i % 2 != 0) else 0
            if self.training and self.use_gc:
                hidden_states, block_entropy = cp.checkpoint(
                    block, hidden_states, shift_size,
                    use_reentrant=False, n_split_index=n_split_index,
                )
            else:
                hidden_states, block_entropy = block(
                    hidden_states, shift_size=shift_size, n_split_index=n_split_index
                )
            gate_entropies.append(block_entropy)

        hidden_states = self.base_model.final_norm(hidden_states)

        loss = None
        logits = None
        
        track_ce = torch.tensor(0.0, device=input_ids.device)
        track_kl = torch.tensor(0.0, device=input_ids.device)
        track_ent = torch.tensor(0.0, device=input_ids.device)

        if labels is not None:
            shift_hidden = hidden_states[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss_fct    = nn.CrossEntropyLoss(reduction="sum", ignore_index=-100)
            total_loss  = 0.0
            valid_tokens = 0.0
            chunk_size  = 2048
            seq_len     = shift_hidden.size(1)

            B = input_ids.size(0)
            teacher_data = []
            if self.training and self.alpha_distill > 0 and logit_shard_id is not None:
                for bi in range(B):
                    is_type_a = (data_type_mask is None) or not data_type_mask[bi].item()
                    if is_type_a:
                        s_id = logit_shard_id[bi].item()
                        s_off = logit_offset[bi].item()
                        L_sample = shift_labels[bi].size(0)
                        t_vals, t_idx = self.logit_cache.get(s_id, s_off, L_sample)
                        if t_vals is not None:
                            teacher_data.append((bi, t_vals, t_idx))
                        else:
                            teacher_data.append((bi, None, None))
                    else:
                        teacher_data.append((bi, None, None))
            else:
                for bi in range(B):
                    teacher_data.append((bi, None, None))

            total_kl_loss = torch.tensor(0.0, device=input_ids.device)
            total_kl_tokens = 0.0

            for i in range(0, seq_len, chunk_size):
                end_idx  = min(i + chunk_size, seq_len)
                c_hidden = shift_hidden[:, i:end_idx, :]
                c_logits = self.base_model.fc_out(c_hidden)
                
                c_logits_scaled = c_logits / 30.0
                c_logits_scaled = torch.tanh(c_logits_scaled) * 30.0
                
                c_labels = shift_labels[:, i:end_idx]

                c_loss   = loss_fct(
                    c_logits_scaled.reshape(-1, c_logits_scaled.size(-1)).float(),
                    c_labels.reshape(-1)
                )
                total_loss   += c_loss
                valid_tokens += (c_labels != -100).sum().item()

                if self.training and self.alpha_distill > 0:
                    for bi, t_vals, t_idx in teacher_data:
                        if t_vals is not None:
                            c_t_vals = t_vals[i:end_idx, :].to(c_logits.device)
                            c_t_idx  = t_idx[i:end_idx, :].to(c_logits.device)
                            c_think  = think_mask[bi, 1+i : 1+end_idx].to(c_logits.device) if think_mask is not None else None
                            
                            kl = sparse_kl_divergence(
                                c_logits_scaled[bi:bi+1],
                                c_t_vals.unsqueeze(0),
                                c_t_idx.unsqueeze(0),
                                temperature=self.distill_temperature,
                                think_mask=c_think.unsqueeze(0) if c_think is not None else None,
                                think_weight=THINK_KL_WEIGHT,
                            )
                            chunk_len = end_idx - i
                            total_kl_loss = total_kl_loss + kl * chunk_len
                            total_kl_tokens += chunk_len

                del c_hidden, c_logits, c_logits_scaled

            loss_ce = total_loss / max(valid_tokens, 1)

            if self.training:
                if gate_entropies:
                    gate_neg_entropy = torch.stack(gate_entropies).mean()
                    loss_entropy = gate_neg_entropy + math.log(3.0)
                else:
                    loss_entropy = torch.tensor(0.0, device=input_ids.device)

                loss_kl = torch.tensor(0.0, device=input_ids.device)
                alpha   = 0.0
                if total_kl_tokens > 0:
                    loss_kl = total_kl_loss / total_kl_tokens
                    alpha = self.alpha_distill

                if data_type_mask is not None:
                    b_indices = data_type_mask.nonzero(as_tuple=True)[0]
                    n_B   = len(b_indices)
                    n_tot = B
                    beta  = self.beta_wave_long if n_B / max(n_tot, 1) > 0.1 else self.beta_wave_short
                else:
                    beta = self.beta_wave_short

                loss = (
                    loss_ce
                    + alpha * loss_kl
                    + beta  * loss_entropy
                    + self.gate_entropy_lambda * loss_entropy
                )
                
                track_ce = loss_ce.detach()
                track_kl = loss_kl.detach() if isinstance(loss_kl, torch.Tensor) else torch.tensor(0.0, device=input_ids.device)
                track_ent = loss_entropy.detach()
            else:
                loss = loss_ce
                track_ce = loss_ce.detach()
        else:
            logits = self.base_model.fc_out(hidden_states)
            logits = logits / 30.0
            logits = torch.tanh(logits) * 30.0

        res = {"loss": loss} if loss is not None else {}
        if logits is not None:
            res["logits"] = logits
        if labels is not None:
            res["ce_loss"] = track_ce
            res["kl_loss"] = track_kl
            res["entropy_loss"] = track_ent
            
        return res


# ============================================================
# 訓練回調
# ============================================================

class DistillMonitor(TrainerCallback):
    def __init__(self, log_path: str = LOG_PATH, save_dir: str = SAVE_DIR):
        self.log_path = log_path
        self.save_dir = save_dir
        self.best_eval_loss = float("inf")
        os.makedirs(save_dir, exist_ok=True)
        if not os.path.exists(log_path):
            with open(log_path, "w", newline="") as f:
                csv.writer(f).writerow(["step", "loss", "ce_loss", "kl_loss", "entropy_loss", "eval_loss", "time"])

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
            
        ce_loss  = logs.get("ce_loss", 0.0)
        kl_loss  = logs.get("kl_loss", 0.0)
        ent_loss = logs.get("entropy_loss", 0.0)
        
        with open(self.log_path, "a", newline="") as f:
            csv.writer(f).writerow([
                state.global_step,
                logs.get("loss", ""),
                f"{ce_loss:.6f}", f"{kl_loss:.6f}", f"{ent_loss:.6f}",
                logs.get("eval_loss", ""),
                time.ctime()
            ])

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not metrics or "eval_loss" not in metrics:
            return
        cur = metrics["eval_loss"]
        if cur < self.best_eval_loss:
            self.best_eval_loss = cur
            model = kwargs.get("model", None)
            if model is not None:
                raw = model.module if hasattr(model, "module") else model
                torch.save(raw.state_dict(), os.path.join(self.save_dir, "best_distill_model.pth"))
                print(f"💾 新最佳 Eval Loss: {cur:.4f}，模型已保存")


class HFUploadCallback(TrainerCallback):
    def __init__(self):
        self._lock = threading.Lock()
        self.best_ce = float("inf")

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not metrics:
            return
        ce = metrics.get("eval_loss", float("inf"))
        if ce < self.best_ce and ce < HF_CE_LOSS_THRESHOLD:
            self.best_ce = ce
            threading.Thread(target=self._upload, args=(state.global_step,), daemon=True).start()

    def _upload(self, step):
        with self._lock:
            try:
                api = HfApi()
                api.create_repo(repo_id=REPO_ID, exist_ok=True, repo_type="model")
                model_path = os.path.join(SAVE_DIR, "best_distill_model.pth")
                if os.path.isfile(model_path) and (step % 2000 == 0 or step <= 200):
                    api.upload_file(
                        path_or_fileobj=model_path,
                        path_in_repo=os.path.basename(model_path),
                        repo_id=REPO_ID, repo_type="model"
                    )
                if os.path.isfile(LOG_PATH):
                    api.upload_file(
                        path_or_fileobj=LOG_PATH,
                        path_in_repo=os.path.basename(LOG_PATH),
                        repo_id=REPO_ID, repo_type="model"
                    )
            except Exception as e:
                print(f"⚠️  HF 上傳失敗：{e}")


class DistillTrainer(Trainer):
    # 已修復 DDP 多卡 Gather 後變成長度為 N 的張量報錯問題
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        if isinstance(outputs, dict) and "loss" in outputs:
            ce = outputs.get("ce_loss", torch.tensor(0.0))
            kl = outputs.get("kl_loss", torch.tensor(0.0))
            ent = outputs.get("entropy_loss", torch.tensor(0.0))
            
            self._last_sub_losses = {
                "ce_loss": ce.mean().item(),
                "kl_loss": kl.mean().item(),
                "entropy_loss": ent.mean().item(),
            }
            
            if outputs["loss"].numel() > 1:
                outputs["loss"] = outputs["loss"].mean()
                
        return (outputs["loss"], outputs) if return_outputs else outputs["loss"]

    def log(self, logs: dict, *args, **kwargs) -> None:
        if hasattr(self, "_last_sub_losses"):
            logs.update(self._last_sub_losses)
        super().log(logs, *args, **kwargs)

    def _save(self, output_dir: str | None = None, state_dict: dict | None = None) -> None:
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        if state_dict is None:
            state_dict = self.model.state_dict()
        
        torch.save(state_dict, os.path.join(output_dir, "pytorch_model.bin"))
        
        if (self.data_collator is not None 
            and hasattr(self.data_collator, "tokenizer") 
            and self.data_collator.tokenizer is not None):
            self.data_collator.tokenizer.save_pretrained(output_dir)
        
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))


# ============================================================
# 主程序
# ============================================================

def main():
    print("🧬 DNA Helix Ultimate 16B Coder 蒸餾訓練引擎啟動...")
    print(f"   Phase {PHASE} | LR={LEARNING_RATE} | Steps={MAX_STEPS}")

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID, trust_remote_code=True)
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    print("\n📦 載入數據集...")
    combined_a_path = os.path.expanduser("~/task06_data/train_type_A_combined.parquet")
    if os.path.exists(combined_a_path):
        a_files = [combined_a_path]
        print(f"   [INFO] Found combined Type A Parquet: {combined_a_path}, loading it directly.")
    else:
        a_files = sorted(glob(os.path.join(DATASET_DIR_A, "*.parquet")))
    b_files = sorted(glob(os.path.join(DATASET_DIR_B, "*.parquet")))

    if not a_files:
        raise FileNotFoundError(f"找不到 Data-Type A Parquet 文件：{DATASET_DIR_A}")
    if not b_files:
        raise FileNotFoundError(f"找不到 Data-Type B Parquet 文件：{DATASET_DIR_B}")

    ds_A_raw = load_dataset("parquet", data_files={"train": a_files}, split="train")
    ds_B_raw = load_dataset("parquet", data_files={"train": b_files}, split="train")

    import glob as glob_module
    available_shards = []
    shard_pattern = os.path.join(LOGIT_SHARD_DIR, "logit_shard_*.safetensors")
    for f in glob_module.glob(shard_pattern):
        if os.path.getsize(f) > 1 * 1024 * 1024 * 1024:
            try:
                shard_id = int(os.path.basename(f).split("_")[2].split(".")[0])
                available_shards.append(shard_id)
            except Exception:
                pass
    
    print(f"   [INFO] Found {len(available_shards)} completed logit shards locally: {sorted(available_shards)}")
    if available_shards:
        indices_to_select = []
        samples_per_shard = 67903
        for s_id in available_shards:
            start_idx = s_id * samples_per_shard
            end_idx = min(start_idx + samples_per_shard, len(ds_A_raw))
            indices_to_select.extend(range(start_idx, end_idx))
            
        ds_A_raw = ds_A_raw.select(indices_to_select)
        print(f"   [INFO] Instantly selected {len(ds_A_raw):,} samples matching available shards.")
    else:
        print(f"   [WARN] No local logit shards found in {LOGIT_SHARD_DIR}!")

    ds_A_split = ds_A_raw.train_test_split(test_size=100, seed=RANDOM_SEED)
    ds_B_split = ds_B_raw.train_test_split(test_size=20,  seed=RANDOM_SEED)

    train_ds = DualStreamDataset(
        ds_A_split["train"], ds_B_split["train"],
        ratio_A=BATCH_MIX_RATIO_A
    )
    eval_ds = DualStreamDataset(
        ds_A_split["test"], ds_B_split["test"],
        ratio_A=BATCH_MIX_RATIO_A
    )

    print(f"   訓練集：{len(train_ds):,} 樣本 | 驗證集：{len(eval_ds):,} 樣本")

    logit_cache = LogitShardCache(LOGIT_SHARD_DIR, n_cache=3)

    print("\n🧬 初始化 16B DNA Helix Ultimate 模型...")
    try:
        from GEMMA3 import DNAHelixUltimate as GEMMA3
        base = GEMMA3(
            vocab_size=151936,
            D=4096,
            C=CHUNK,
            hidden_dim=16384,
            num_blocks=40,
            rope_local=10000.0,
            rope_global=1000000.0,
        )
        print(f"   模型參數量：{sum(p.numel() for p in base.parameters())/1e9:.1f}B")
    except ImportError:
        raise ImportError("請確保 GEMMA3.py 存在且可導入（16B 版本）")

    model = DNAHelixDistillLM(
        base_model=base,
        logit_shard_cache=logit_cache,
        use_gc=True,
        gate_entropy_lambda=GATE_ENTROPY_LAMBDA,
        distill_temperature=DISTILL_TEMPERATURE,
        alpha_distill=ALPHA_DISTILL,
        beta_wave_long=BETA_WAVE_LONG,
        beta_wave_short=BETA_WAVE_SHORT,
    )

    for p in model.parameters():
        p.requires_grad = True

    model = model.cuda().to(torch.bfloat16)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   ✅ 可訓練參數：{total_params/1e9:.2f}B")

    training_args = TrainingArguments(
        output_dir                  = SAVE_DIR,
        max_steps                   = MAX_STEPS,
        per_device_train_batch_size = BATCH_SIZE_PER_DEVICE,
        per_device_eval_batch_size  = BATCH_SIZE_PER_DEVICE,
        gradient_accumulation_steps = GRAD_ACCUMULATION_STEPS,
        logging_steps               = LOGGING_STEPS,
        learning_rate               = LEARNING_RATE,
        lr_scheduler_type           = "cosine",
        warmup_steps                = WARMUP_STEPS,
        bf16                        = True,
        eval_strategy               = "steps",
        eval_steps                  = EVAL_STEPS,
        save_steps                  = SAVE_STEPS,
        save_total_limit            = SAVE_TOTAL_LIMIT,
        optim                       = "paged_adamw_8bit",
        remove_unused_columns       = False,
        report_to                   = "none",
        dataloader_num_workers      = 4,
        dataloader_pin_memory       = True,
    )

    collator = DualStreamCollator(pad_token_id=pad_id)

    trainer = DistillTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        callbacks=[
            DistillMonitor(log_path=LOG_PATH, save_dir=SAVE_DIR),
            HFUploadCallback(),
        ],
    )

    last_ckpt = get_last_checkpoint(SAVE_DIR)
    if last_ckpt:
        print(f"\n🔄 從斷點恢復訓練：{last_ckpt}")
        trainer.train(resume_from_checkpoint=last_ckpt)
    else:
        print(f"\n🚀 開始 Phase {PHASE} 訓練...")
        trainer.train()

    final_path = os.path.join(SAVE_DIR, f"dna16b_phase{PHASE}_final.pth")
    raw_model  = model.module if hasattr(model, "module") else model
    torch.save(raw_model.state_dict(), final_path)
    print(f"\n✅ Phase {PHASE} 訓練完成！模型已保存至：{final_path}")


if __name__ == "__main__":
    main()