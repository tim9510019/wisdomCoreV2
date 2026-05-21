# trainCPTAC_AGI_GEMMA3_1K_DYNX.py — AGI GEMMA 3X (頻域下採樣降維 + DYNAMIC 模式) 實驗
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

# 🌟 引入全新設計的 AGIGEMMA3X
from AGIGEMMA3X import AGIGEMMA3X, AGIGEMMA3XForCausalLM
from utils import transplant_and_freeze

RANDOM_SEED = 2026
set_seed(RANDOM_SEED)

# ==========================================
# [ 全局配置區 ] — 參數已統一放置於此
# ==========================================
# 🌟 下採樣/KV壓縮倍率 (2 的倍數，例如 2, 4, 8)
DOWNSAMPLE_FACTOR = 2

# 🌟 FFT 濾波器層數配置
N_FFT_F = 6                 # 前置 FFT 層數
N_FFT_B = 4                 # 後置 FFT 層數 (4 層將 KV 長度變回原本的長度)

# 🌟 其他網路與訓練配置參數
MODEL_ID = "google/gemma-3-1b-it"
DATASET_DIR = "~/agiv2_stage1_1K"
SAVE_DIR = "~/agigemma3_cpt_checkpoints_1K_DYNX"
LOG_PATH = "~/agigemma3_cpt_1k_DYNX_log.csv"

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
LEARNING_RATE = 1e-5
CHUNK = 256
ROPE_LOCAL = 10000.0
ROPE_GLOBAL = 1000000.0

# HuggingFace 自動上傳配置
REPO_ID = "tim9510019/AGIGEMMA3-1B-CPT_1K_DYNX"
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
# [ 量子退火監控器 (無閘門精簡版) ]
# ==========================================
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
                        if len(row) > 3 and row[3].strip():
                            try:
                                val = float(row[3])
                                if val < self.best_eval_loss:
                                    self.best_eval_loss = val
                            except ValueError:
                                pass
                if self.best_eval_loss != float("inf"):
                    print(
                        f"\n📈 [Monitor] 當前最佳 Eval Loss 基準: {self.best_eval_loss:.4f}"
                    )
            except Exception as e:
                print(f"\n⚠️ [Monitor] 讀取歷史紀錄失敗: {e}")
        else:
            with open(self.path, "w", newline="") as f:
                csv.writer(f).writerow(["step", "loss", "ce_loss", "eval_loss", "time"])

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            model = kwargs.get("model", None)
            ce_loss_val = ""
            if model is not None:
                raw_model = model.module if hasattr(model, "module") else model
                if hasattr(raw_model, "_last_ce_loss"):
                    ce_loss_val = f"{raw_model._last_ce_loss:.6f}"
                    logs["ce_loss"] = raw_model._last_ce_loss

            with open(self.path, "a", newline="") as f:
                csv.writer(f).writerow(
                    [
                        state.global_step,
                        logs.get("loss", ""),
                        ce_loss_val,
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
                else:
                    print(
                        f"\n[HF Upload] ⚠️ 找不到模型檔案 {self.model_path}，跳過模型上傳。"
                    )

                if os.path.isfile(self.log_path):
                    api.upload_file(
                        path_or_fileobj=self.log_path,
                        path_in_repo=os.path.basename(self.log_path),
                        repo_id=self.repo_id,
                        repo_type="model",
                        commit_message=f"[Auto] log update step={step}",
                    )
                    print(f"[HF Upload] ✅ CSV Log 上傳完成")
                else:
                    print(
                        f"[HF Upload] ⚠️ 找不到 CSV Log {self.log_path}，跳過 Log 上傳。"
                    )

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
            print(
                f"\n[HF Upload] 🚀 CE Loss 突破門檻 ({ce_loss:.4f} < {self.ce_loss_threshold})，啟動背景上傳..."
            )
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
# [ 核心引擎 ]
# ==========================================
def main():
    print(
        f"\n🚀 啟動 AGIGEMMA3X CPT 訓練矩陣 [DYNX 模式] (前 {N_FFT_F} 後 {N_FFT_B} 層 DynamicPhaseLockFFT + Gemma3 {DOWNSAMPLE_FACTOR}倍降維與權重凍結)..."
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

    # ==========================================
    # 🌟 核心替換：採用全新的 AGIGEMMA3X 降維模型
    # ==========================================
    base = AGIGEMMA3X(
        vocab_size=262144,
        D=1152,
        C=CHUNK,
        hidden_dim=6912,
        num_blocks=26,
        N_fft_F=N_FFT_F,
        N_fft_B=N_FFT_B,
        rope_local=ROPE_LOCAL,
        rope_global=ROPE_GLOBAL,
        fft_block_type="dynamic",           # 🌟 DYNX 模式使用 dynamic
        downsample_factor=DOWNSAMPLE_FACTOR # 🌟 設定下採樣倍率
    )

    print("\n🔄 正在掛載原生 google/gemma-3-1b-it 權重並凍結主幹...")
    base = transplant_and_freeze(MODEL_ID, base)

    # 使用 AGI GEMMA 3X 專屬 Wrapper
    model = AGIGEMMA3XForCausalLM(base, use_gc=True)
    print("✅ 模型實例化完成，並成功接合官方預訓練特徵與頻域下採樣物理機制。")

    # 🔒 Gemma3 主幹凍結，只訓練 FFT blocks 與頻域變換層
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"🔒 Gemma3 凍結參數: {frozen_params:,} | 🟢 FFT與縮放新增層可訓練參數: {trainable_params:,}")

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
            ThermalControlCallback(delay_seconds=0.0),  # 🛡️ 植入硬體保護控制器
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


if __name__ == "__main__":
    main()
