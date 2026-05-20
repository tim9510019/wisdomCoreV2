# trainCPTAC_1KSGS_NV.py — AGIV2 持續預訓練終極引擎 (動態 MoE 路由版 + 1.5秒主動式散熱)
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

from AGIV2GACGS import AGIV2G
from utils import AGIV2GForCausalLMT

RANDOM_SEED = 2026
set_seed(RANDOM_SEED)

# ==========================================
# [ 全局配置區 ]
# ==========================================
MODEL_ID = "google/gemma-3-1b-it"
DATASET_DIR = "~/agiv2_stage1_1K"
SAVE_DIR = "~/agiv2_cpt_checkpoints_1KSGS_NV"
LOG_PATH = "~/agiv2_cpt_1KSGS_NV_log.csv"
BEST_ROUTER_32K_PATH = "~/agiv2_zerogate_ac_checkpoints_32KSGT/best_model.pth"

DATASET_DIR = os.path.expanduser(DATASET_DIR)
SAVE_DIR = os.path.expanduser(SAVE_DIR)
LOG_PATH = os.path.expanduser(LOG_PATH)
BEST_ROUTER_32K_PATH = os.path.expanduser(BEST_ROUTER_32K_PATH)

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
GATE_ENTROPY_LAMBDA = 0.01  # 閘門負熵正則化權重 λ
ROPE_LOCAL = 10000.0
ROPE_GLOBAL = 10000.0

# HuggingFace 自動上傳配置
REPO_ID = "tim9510019/AGIV2-1B-CPT_GS_NV"
HF_CE_LOSS_THRESHOLD = 13.0  # 只有 ce_loss 低於此值才觸發上傳


# ==========================================
# [ 硬體保護：主動式散熱控制器 ]
# ==========================================
class ThermalControlCallback(TrainerCallback):
    def __init__(self, delay_seconds=1.5):
        self.delay_seconds = delay_seconds

    def on_step_end(self, args, state, control, **kwargs):
        time.sleep(self.delay_seconds)


# ==========================================
# [ 量子退火監控器 (動態閘門觀測版) ]
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
                        if len(row) > 4 and row[4].strip():
                            try:
                                val = float(row[4])
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
                # CSV 標題改為記錄兩個動態閘門的分量
                csv.writer(f).writerow(
                    [
                        "step",
                        "loss",
                        "ce_loss",
                        "gate_ent_loss",
                        "eval_loss",
                        "g_loc",
                        "g_mem",
                        "time",
                    ]
                )

    def on_step_begin(self, args, state, control, **kwargs):
        model = kwargs.get("model", None)
        if model is not None:
            # 溫度從 2.0 線性降至 0.5
            progress = state.global_step / max(1, state.max_steps)
            current_temp = max(0.5, 2.0 - 1.5 * progress)

            raw_model = model.module if hasattr(model, "module") else model
            core_model = (
                raw_model.base_model if hasattr(raw_model, "base_model") else raw_model
            )
            if hasattr(core_model, "set_temperature"):
                core_model.set_temperature(current_temp)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            model = kwargs.get("model", None)
            g_loc_avg, g_mem_avg = 0.0, 0.0
            blocks_count = 0

            # 🌟 攔截並計算所有 Global Block 的閘門平均值
            if model is not None:
                raw_model = model.module if hasattr(model, "module") else model
                core_model = (
                    raw_model.base_model
                    if hasattr(raw_model, "base_model")
                    else raw_model
                )

                for b in core_model.blocks:
                    if hasattr(b, "avg_g_loc"):
                        g_loc_avg += b.avg_g_loc.item()
                        g_mem_avg += b.avg_g_mem.item()
                        blocks_count += 1

            if blocks_count > 0:
                g_loc_avg /= blocks_count
                g_mem_avg /= blocks_count

            # 更新 logs 結構以供後續 callback 使用
            logs["g_loc"] = round(g_loc_avg, 4)
            logs["g_mem"] = round(g_mem_avg, 4)

            # 🌟 從模型側通道讀取分離的 CE loss 與閘門負熵損失
            ce_loss_val = ""
            gate_ent_val = ""
            if model is not None:
                raw_model = model.module if hasattr(model, "module") else model
                if hasattr(raw_model, "_last_ce_loss"):
                    ce_loss_val = f"{raw_model._last_ce_loss:.6f}"
                    logs["ce_loss"] = raw_model._last_ce_loss
                if hasattr(raw_model, "_last_gate_entropy_loss"):
                    gate_ent_val = f"{raw_model._last_gate_entropy_loss:.6f}"
                    logs["gate_ent_loss"] = raw_model._last_gate_entropy_loss

            with open(self.path, "a", newline="") as f:
                csv.writer(f).writerow(
                    [
                        state.global_step,
                        logs.get("loss", ""),
                        ce_loss_val,
                        gate_ent_val,
                        logs.get("eval_loss", ""),
                        f"{g_loc_avg:.4f}",
                        f"{g_mem_avg:.4f}",
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
    """當 ce_loss 創新低且低於門檻值時，於背景執行緒自動上傳 Best Model 與 CSV。"""

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
        self._upload_lock = threading.Lock()  # 防止重疊上傳

    def _do_upload(self, ce_loss_val, step):
        """於背景執行緒執行實際上傳動作。"""
        with self._upload_lock:
            try:
                api = HfApi()
                api.create_repo(repo_id=self.repo_id, exist_ok=True, repo_type="model")

                # 上傳 Best Model
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

                # 上傳 CSV Log
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

        # eval_loss 現在是純 CE loss（eval 時不加 entropy 懲罰）
        ce_loss = metrics.get("eval_loss", None)
        if ce_loss is None:
            return

        # 觸發條件：ce_loss 創新低 且 低於門檻值
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
    print("\n🚀 啟動 AGIV2 CPT 第一階訓練矩陣 (動態 MoE 路由版)...")

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

    base = AGIV2G(
        vocab_size=262144,
        D=1152,
        C=CHUNK,
        hidden_dim=6912,
        num_blocks=26,
        rope_local=ROPE_LOCAL,
        rope_global=ROPE_GLOBAL,
    )
    model = AGIV2GForCausalLMT(
        base, use_gc=True, gate_entropy_lambda=GATE_ENTROPY_LAMBDA
    )
    print("✅ 模型實例化完成，採用原生隨機初始化。")

    if os.path.exists(BEST_ROUTER_32K_PATH):
        print(f"\n🔄 載入 32K 階段基礎權重: {BEST_ROUTER_32K_PATH}")
        model.load_state_dict(
            torch.load(BEST_ROUTER_32K_PATH, map_location="cpu"), strict=False
        )
        print("✅ 路由權重載入完成。")
    else:
        print(
            f"\n⚠️ 找不到 32K 路由權重檔案 {BEST_ROUTER_32K_PATH}，將完全從零開始訓練！"
        )

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
            ThermalControlCallback(delay_seconds=0.0),  # 🛡️ 植入 1.5 秒硬體保護控制器
            HFAutoUploadCallback(),  # 🌐 CE Loss 突破門檻自動上傳至 HuggingFace
        ],
    )

    last_checkpoint = get_last_checkpoint(SAVE_DIR)
    trainer.train(resume_from_checkpoint=last_checkpoint)
    torch.save(
        (
            model.state_dict()
            if not hasattr(model, "module")
            else model.module.state_dict()
        ),
        os.path.join(SAVE_DIR, "final_agiv2_cpt_1k.pth"),
    )


if __name__ == "__main__":
    main()
