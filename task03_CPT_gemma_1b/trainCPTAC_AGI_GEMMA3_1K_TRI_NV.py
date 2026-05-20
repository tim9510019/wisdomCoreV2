# trainCPTAC_AGI_GEMMA3_1K_TRI_NV.py — AGI GEMMA 3 (正三角形幾何約束版)
import os
import sys
import csv
import time
import threading

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["BITSANDBYTES_NOWELCOME"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import math
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

from AGIGEMMA3 import AGIGEMMA3
from utils import transplant_and_freeze

RANDOM_SEED = 2026
set_seed(RANDOM_SEED)

# ==========================================
# [ 全局配置區 ]
# ==========================================
N_FFT_F = 6
N_FFT_B = 4

MODEL_ID = "google/gemma-3-1b-it"
DATASET_DIR = "~/agiv2_stage1_1K"
SAVE_DIR = "~/agigemma3_cpt_checkpoints_1K_TRI_NV"
LOG_PATH = "~/agigemma3_cpt_1k_TRI_NV_log.csv"

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
ROPE_LOCAL = 10000.0
ROPE_GLOBAL = 1000000.0

# HuggingFace 自動上傳配置
REPO_ID = "tim9510019/AGIGEMMA3-1B-CPT_1K_TRI_NV"
HF_CE_LOSS_THRESHOLD = 2.65


# ==========================================
# [ 硬體保護：主動式散熱控制器 ]
# ==========================================
class ThermalControlCallback(TrainerCallback):
    def __init__(self, delay_seconds=1.5):
        self.delay_seconds = delay_seconds

    def on_step_end(self, args, state, control, **kwargs):
        time.sleep(self.delay_seconds)


# ==========================================
# [ 幾何退火與監控器 ]
# ==========================================
class GeometricAnnealingCallback(TrainerCallback):
    def __init__(self, max_steps):
        self.max_steps = max_steps
        self.warmup_steps = int(max_steps * 0.1)
        self.crystallize_steps = int(max_steps * 0.5)

    def on_step_begin(self, args, state, control, **kwargs):
        model = kwargs.get("model", None)
        if model is not None:
            raw_model = model.module if hasattr(model, "module") else model
            step = state.global_step

            if step < self.warmup_steps:
                weight = 0.01
            elif step < self.warmup_steps + self.crystallize_steps:
                progress = (step - self.warmup_steps) / self.crystallize_steps
                weight = 0.01 + progress * (0.5 - 0.01)
            else:
                weight = 0.5

            if hasattr(raw_model, "geom_weight"):
                raw_model.geom_weight.fill_(weight)


class QuantumCPTMonitor(TrainerCallback):
    def __init__(self, path=LOG_PATH, save_dir=SAVE_DIR):
        self.path = path
        self.save_dir = save_dir
        self.best_eval_loss = float("inf")
        os.makedirs(self.save_dir, exist_ok=True)

        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    reader = csv.reader(f)
                    next(reader, None)
                    for row in reader:
                        if len(row) > 5 and row[5].strip():
                            try:
                                val = float(row[5])
                                if val < self.best_eval_loss:
                                    self.best_eval_loss = val
                            except ValueError:
                                pass
            except Exception as e:
                print(f"\n⚠️ [Monitor] 讀取歷史紀錄失敗: {e}")
        else:
            with open(self.path, "w", newline="") as f:
                csv.writer(f).writerow(
                    [
                        "step",
                        "loss",
                        "ce_loss",
                        "geom_loss",
                        "tri_loss",
                        "eval_loss",
                        "time",
                    ]
                )

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            model = kwargs.get("model", None)
            ce_loss_val = ""
            geom_loss_val = ""
            tri_loss_val = ""

            if model is not None:
                raw_model = model.module if hasattr(model, "module") else model
                if hasattr(raw_model, "_last_ce_loss"):
                    ce_loss_val = f"{raw_model._last_ce_loss:.6f}"
                    logs["ce_loss"] = raw_model._last_ce_loss
                if hasattr(raw_model, "_last_geom_loss"):
                    geom_loss_val = f"{raw_model._last_geom_loss:.6f}"
                    logs["geom_loss"] = raw_model._last_geom_loss
                if hasattr(raw_model, "_last_tri_loss"):
                    tri_loss_val = f"{raw_model._last_tri_loss:.6f}"
                    logs["tri_loss"] = raw_model._last_tri_loss

            with open(self.path, "a", newline="") as f:
                csv.writer(f).writerow(
                    [
                        state.global_step,
                        logs.get("loss", ""),
                        ce_loss_val,
                        geom_loss_val,
                        tri_loss_val,
                        logs.get("eval_loss", ""),
                        time.ctime(),
                    ]
                )

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics and "eval_loss" in metrics:
            current_eval_loss = metrics["eval_loss"]
            if current_eval_loss < self.best_eval_loss:
                old_best = self.best_eval_loss
                self.best_eval_loss = current_eval_loss
                model = kwargs.get("model", None)
                if model is not None:
                    best_model_path = os.path.join(self.save_dir, "best_cpt_model.pth")
                    raw_model = model.module if hasattr(model, "module") else model
                    torch.save(raw_model.state_dict(), best_model_path)
                    print(
                        f"\n[Monitor] 🌟 收斂突破 ({old_best:.4f} -> {current_eval_loss:.4f})，儲存至 {best_model_path}"
                    )
            else:
                print(
                    f"\n[Monitor] 🛡️ 此次成績 ({current_eval_loss:.4f}) 未超越歷史最佳 ({self.best_eval_loss:.4f})。"
                )


# ==========================================
# [ HuggingFace 自動上傳器 ]
# ==========================================
class HFAutoUploadCallback(TrainerCallback):
    def __init__(
        self,
        repo_id=REPO_ID,
        model_path=None,
        log_path=LOG_PATH,
        ce_loss_threshold=HF_CE_LOSS_THRESHOLD,
        save_dir=SAVE_DIR,
    ):
        self.repo_id = repo_id
        self.model_path = model_path or os.path.join(save_dir, "best_cpt_model.pth")
        self.log_path = log_path
        self.ce_loss_threshold = ce_loss_threshold
        self.best_ce_loss = float("inf")
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
                        commit_message=f"[Auto] step={step} ce_loss={ce_loss_val:.6f}",
                    )
                    print(
                        f"\n[HF Upload] ✅ Best Model 上傳完成 (step={step}, ce_loss={ce_loss_val:.4f})"
                    )

                if os.path.isfile(self.log_path):
                    api.upload_file(
                        path_or_fileobj=self.log_path,
                        path_in_repo=os.path.basename(self.log_path),
                        repo_id=self.repo_id,
                        repo_type="model",
                        commit_message=f"[Auto] log update step={step}",
                    )
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
            t = threading.Thread(
                target=self._do_upload, args=(ce_loss, step), daemon=True
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
            "n_split_index": torch.tensor(batch_n_splits, dtype=torch.long),
        }


# ==========================================
# 幾何三頭投影模組
# ==========================================
class TriHeadProjection(nn.Module):
    def __init__(self, D):
        super().__init__()
        # Head A: 負責原本的語言輸出
        self.head_A = nn.Sequential(nn.Linear(D, D * 2), nn.GELU(), nn.Linear(D * 2, D))
        # Head A_反: 負責反向隱性特徵
        self.head_A_anti = nn.Sequential(
            nn.Linear(D, D * 2), nn.GELU(), nn.Linear(D * 2, D)
        )
        # Head A_超: 負責超然視角特徵
        self.head_A_super = nn.Sequential(
            nn.Linear(D, D * 2), nn.GELU(), nn.Linear(D * 2, D)
        )

        # 初始狀態下 A 幾乎等於 h_t，保證語言能力不崩潰
        nn.init.zeros_(self.head_A[-1].weight)
        if self.head_A[-1].bias is not None:
            nn.init.zeros_(self.head_A[-1].bias)

    def forward(self, h):
        A = h + self.head_A(h)
        A_anti = self.head_A_anti(h)
        A_super = self.head_A_super(h)
        return A, A_anti, A_super


# ==========================================
# 專屬 AGIGEMMA3 的 TRI Wrapper
# 負責在最後加上三向投影，並強制收斂至正三角形幾何
# ==========================================
class AGIGEMMA3ForCausalLM_TRI(nn.Module):
    def __init__(self, base_model, use_gc=True):
        super().__init__()
        self.base_model = base_model
        self.use_gc = use_gc

        D = self.base_model.D
        self.tri_head = TriHeadProjection(D)

        # 用於退火的幾何 Loss 權重，初始為 0.01
        self.register_buffer("geom_weight", torch.tensor(0.01))

    def forward(self, input_ids, labels=None, **kwargs):
        n_split_index = kwargs.get("n_split_index", None)

        hidden_states = self.base_model.embedding(input_ids)
        hidden_states = hidden_states * math.sqrt(self.base_model.D)

        # 1. 執行前 N 層 FFT
        for block in self.base_model.pre_fft_blocks:
            if self.training and self.use_gc:
                hidden_states = checkpoint.checkpoint(
                    block, hidden_states, 0, n_split_index, use_reentrant=False
                )
            else:
                hidden_states = block(
                    hidden_states, shift_size=0, n_split_index=n_split_index
                )

        # 2. 執行原 Gemma3 的 26 層
        for i, block in enumerate(self.base_model.blocks):
            shift_size = (block.C // 2) if (i % 2 != 0) else 0
            if self.training and self.use_gc:
                hidden_states = checkpoint.checkpoint(
                    block, hidden_states, shift_size, n_split_index, use_reentrant=False
                )
            else:
                hidden_states = block(
                    hidden_states, shift_size=shift_size, n_split_index=n_split_index
                )

        # 3. 執行後 N 層 FFT
        for block in self.base_model.post_fft_blocks:
            if self.training and self.use_gc:
                hidden_states = checkpoint.checkpoint(
                    block, hidden_states, 0, n_split_index, use_reentrant=False
                )
            else:
                hidden_states = block(
                    hidden_states, shift_size=0, n_split_index=n_split_index
                )

        hidden_states = self.base_model.final_norm(hidden_states)

        # --- 幾何三頭分流 ---
        A, A_anti, A_super = self.tri_head(hidden_states)

        loss = None
        logits = None

        if labels is not None:
            shift_A = A[..., :-1, :].contiguous()
            shift_A_anti = A_anti[..., :-1, :].contiguous()
            shift_A_super = A_super[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss_fct = nn.CrossEntropyLoss(reduction="sum", ignore_index=-100)
            total_ce_loss = 0.0
            valid_tokens = 0.0
            chunk_size = 2048
            seq_len = shift_A.size(1)

            for i in range(0, seq_len, chunk_size):
                end_idx = min(i + chunk_size, seq_len)
                c_A = shift_A[:, i:end_idx, :]
                c_logits = self.base_model.fc_out(c_A)
                c_labels = shift_labels[:, i:end_idx]

                c_loss = loss_fct(
                    c_logits.reshape(-1, c_logits.size(-1)).float(),
                    c_labels.reshape(-1),
                )
                total_ce_loss += c_loss

                valid_tokens += (c_labels != -100).sum().item()
                del c_logits, c_A

            ce_loss = total_ce_loss / max(valid_tokens, 1)

            if self.training:
                valid_mask = (shift_labels != -100).unsqueeze(-1)  # (B, L, 1)

                # 對特徵進行 L2 正規化 (長度為 1)
                norm_A = F.normalize(shift_A, p=2, dim=-1)
                norm_A_anti = F.normalize(shift_A_anti, p=2, dim=-1)
                norm_A_super = F.normalize(shift_A_super, p=2, dim=-1)

                # 計算三角張力 Loss (兩兩夾角 -0.5)
                cos_A_anti = (norm_A * norm_A_anti).sum(dim=-1)
                cos_anti_super = (norm_A_anti * norm_A_super).sum(dim=-1)
                cos_super_A = (norm_A_super * norm_A).sum(dim=-1)
                loss_tri_element = (
                    (cos_A_anti + 0.5) ** 2
                    + (cos_anti_super + 0.5) ** 2
                    + (cos_super_A + 0.5) ** 2
                )

                # 計算質心歸零力 Loss (能量守恆)
                centroid = norm_A + norm_A_anti + norm_A_super
                loss_zero_element = (centroid**2).sum(dim=-1)

                # 僅對有效 Target 計算幾何 Loss
                valid_mask_sq = valid_mask.squeeze(-1)
                loss_tri_valid = (loss_tri_element * valid_mask_sq).sum() / max(
                    valid_tokens, 1
                )
                loss_zero_valid = (loss_zero_element * valid_mask_sq).sum() / max(
                    valid_tokens, 1
                )

                geom_loss = loss_tri_valid + loss_zero_valid

                # 結合 Language CE Loss 與 幾何 Loss
                loss = ce_loss + self.geom_weight * geom_loss

                self._last_ce_loss = ce_loss.detach().item()
                self._last_geom_loss = geom_loss.detach().item()
                self._last_tri_loss = loss_tri_valid.detach().item()
            else:
                loss = ce_loss
        else:
            # 推理時，預設使用 A (顯性能量頭) 作為語言輸出
            logits = self.base_model.fc_out(A)

        return (
            {"loss": loss, "logits": logits} if logits is not None else {"loss": loss}
        )


# ==========================================
# [ 核心引擎 ]
# ==========================================
def main():
    print(
        f"\n🚀 啟動 AGIGEMMA3 CPT 訓練矩陣 (前 {N_FFT_F} 後 {N_FFT_B} 層純 FFT + 正三角形幾何約束)..."
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    parquet_file = os.path.join(DATASET_DIR, "agiv2_stage1_N_B.parquet")
    if not os.path.exists(parquet_file):
        raise FileNotFoundError(
            f"找不到實體矩陣 {parquet_file}，請先執行 cpt1K.py 產生資料。"
        )

    print(f"📦 成功鎖定實體磁區，直接掛載 Parquet 拓撲: {parquet_file}")
    dataset_dict = load_dataset("parquet", data_files=parquet_file)

    dataset = dataset_dict["train"].train_test_split(test_size=50, seed=RANDOM_SEED)
    train_ds, eval_ds = dataset["train"], dataset["test"]
    print(
        f"✅ 資料映射與絕對隔離完成！訓練集: {len(train_ds):,} 筆 | 獨立驗證集: {len(eval_ds)} 筆"
    )

    # 初始化 AGIGEMMA3 架構
    base = AGIGEMMA3(
        vocab_size=262144,
        D=1152,
        C=CHUNK,
        hidden_dim=6912,
        num_blocks=26,
        N_fft_F=N_FFT_F,
        N_fft_B=N_FFT_B,
        rope_local=ROPE_LOCAL,
        rope_global=ROPE_GLOBAL,
    )

    print("\n🔄 正在掛載原生 google/gemma-3-1b-it 權重...")
    base = transplant_and_freeze(MODEL_ID, base)

    # 套用全新具備三頭幾何對齊與退火能力的 Wrapper
    model = AGIGEMMA3ForCausalLM_TRI(base, use_gc=True)
    print("✅ 幾何投影頭 (Tri-Head) 實例化完成。")

    print("\n🔧 啟動全參數訓練設定:")
    for param in model.parameters():
        param.requires_grad = True

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✅ 設定完成。可訓練參數總量 (全參數微調): {trainable_params:,}")

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
            GeometricAnnealingCallback(max_steps=MAX_STEPS),  # 引入熱力學退火
            QuantumCPTMonitor(path=LOG_PATH, save_dir=SAVE_DIR),
            ThermalControlCallback(delay_seconds=0.0),
            HFAutoUploadCallback(),
        ],
    )

    last_checkpoint = get_last_checkpoint(SAVE_DIR)
    if last_checkpoint is not None:
        print(f"\n🚀 偵測到時間錨點 {last_checkpoint}，啟動狀態無損恢復...")
        trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        print("\n🚀 開始全新訓練...")
        trainer.train()

    torch.save(
        (
            model.state_dict()
            if not hasattr(model, "module")
            else model.module.state_dict()
        ),
        os.path.join(SAVE_DIR, "final_agigemma3_cpt_1k_tri.pth"),
    )


if __name__ == "__main__":
    main()
