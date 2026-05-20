# trainCPTAC_1K.py — AGIV2 持續預訓練終極引擎 (From Scratch 全參數訓練版)
import os
import sys
import csv
import time
import torch
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    set_seed,
    TrainerCallback,
)
from transformers.trainer_utils import get_last_checkpoint
from datasets import load_dataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["BITSANDBYTES_NOWELCOME"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from AGIV2GAC import AGIV2G
from utils import AGIV2GForCausalLM  # 移除了 transplant_and_freeze

RANDOM_SEED = 2026
set_seed(RANDOM_SEED)

# ==========================================
# [ 全局配置區 ]
# ==========================================
MODEL_ID = "google/gemma-3-1b-it"  # 僅用於載入 Tokenizer
DATASET_DIR = "./agiv2_stage1_1K"
SAVE_DIR = "./agiv2_cpt_checkpoints_1KS"
LOG_PATH = "./agiv2_cpt_1kS_log.csv"

BEST_ROUTER_32K_PATH = "./agiv2_zerogate_ac_checkpoints_32KS/best_model.pth"

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


# ==========================================
# [ 統一對齊的量子監控器 (CPT 版) ]
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
                        if len(row) > 2 and row[2].strip():
                            try:
                                val = float(row[2])
                                if val < self.best_eval_loss:
                                    self.best_eval_loss = val
                            except ValueError:
                                pass
                if self.best_eval_loss != float("inf"):
                    print(
                        f"\n📈 [Monitor] 時間線同步成功。當前最佳 Eval Loss 基準: {self.best_eval_loss:.4f}"
                    )
            except Exception as e:
                print(f"\n⚠️ [Monitor] 讀取歷史紀錄失敗: {e}")
        else:
            with open(self.path, "w", newline="") as f:
                csv.writer(f).writerow(
                    ["step", "loss", "eval_loss", "fft_max", "mem_max", "time"]
                )

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            model = kwargs.get("model", None)
            max_fft, max_mem = 0.0, 0.0
            if model is not None:
                raw_model = model.module if hasattr(model, "module") else model
                core_model = (
                    raw_model.base_model
                    if hasattr(raw_model, "base_model")
                    else raw_model
                )
                g_fft = [
                    b.gate_fft.item()
                    for b in core_model.blocks
                    if hasattr(b, "gate_fft")
                ]
                g_mem = [
                    b.gate_mem.item()
                    for b in core_model.blocks
                    if hasattr(b, "gate_mem")
                ]
                if g_fft:
                    max_fft, max_mem = max(g_fft), max(g_mem)
                    logs["fft_max"] = round(max_fft, 6)
                    logs["mem_max"] = round(max_mem, 6)

            with open(self.path, "a", newline="") as f:
                csv.writer(f).writerow(
                    [
                        state.global_step,
                        logs.get("loss", ""),
                        logs.get("eval_loss", ""),
                        max_fft,
                        max_mem,
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
                        f"\n[Monitor] 🌟 結構極化收斂，發現更佳權重 ({old_best:.4f} -> {current_eval_loss:.4f})，已儲存至 {best_model_path}"
                    )
            else:
                print(
                    f"\n[Monitor] 🛡️ 此次成績 ({current_eval_loss:.4f}) 未超越歷史最佳 ({self.best_eval_loss:.4f})，跳過保存。"
                )


# ==========================================
# [ 物理隔離 ] N -> N+B 絕對斷層拼接器 (非對稱解耦版)
# ==========================================
class CPTDataCollator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, features):
        batch_input_ids = []
        batch_labels = []
        batch_n_splits = []

        max_len = max(len(f["input_ids"]) for f in features)

        for f in features:
            seq = f["input_ids"]
            n_split = f["n_split_index"]
            n_split = min(n_split, len(seq))

            pad_len = max_len - len(seq)
            padded_input = seq + [self.pad_token_id] * pad_len
            batch_input_ids.append(padded_input)

            labels = [-100] * n_split + seq[n_split:] + [-100] * pad_len
            batch_labels.append(labels)

            batch_n_splits.append(n_split)

        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
            "n_split_index": torch.tensor(batch_n_splits, dtype=torch.long),
        }


# ==========================================
# [ 核心引擎 ] 主執行緒與全參數解凍
# ==========================================
def main():
    print("\n🚀 啟動 AGIV2 CPT 第一階訓練矩陣 (From Scratch 全參數訓練版)...")

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
    train_ds = dataset["train"]
    eval_ds = dataset["test"]
    print(
        f"✅ 資料映射與絕對隔離完成！訓練集: {len(train_ds):,} 筆 | 獨立驗證集: {len(eval_ds)} 筆"
    )

    # 🌟 核心修改點 1：直接實例化，不移植 Gemma 3 參數
    base = AGIV2G(vocab_size=262144, D=1152, C=CHUNK, hidden_dim=6912, num_blocks=26)
    model = AGIV2GForCausalLM(base, use_gc=True)
    print("✅ 模型實例化完成，採用原生隨機初始化。")

    if os.path.exists(BEST_ROUTER_32K_PATH):
        print(f"\n🔄 成功尋獲 32K 階段之最佳路由權重，嘗試載入: {BEST_ROUTER_32K_PATH}")
        state_dict = torch.load(BEST_ROUTER_32K_PATH, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
        print("✅ 路由權重載入完成。")
    else:
        print(
            f"\n⚠️ 找不到 32K 路由權重檔案 {BEST_ROUTER_32K_PATH}，將完全從零開始訓練！"
        )

    # 強制拓撲重置 (Zero-Gate Initialization)
    print("\n🔧 執行強制拓撲重置：將所有 gate_fft 與 gate_mem 初始化為 0.0...")
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "gate_fft" in name or "gate_mem" in name:
                param.data.fill_(1.0)
    print("✅ 閘門已全數歸零。")

    # 🌟 核心修改點 2：全參數解凍 (From Scratch 必備)
    print("\n🔧 啟動全參數訓練設定:")
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
        callbacks=[QuantumCPTMonitor(path=LOG_PATH, save_dir=SAVE_DIR)],
    )

    last_checkpoint = get_last_checkpoint(SAVE_DIR)
    if last_checkpoint is not None:
        print(f"\n🚀 偵測到時間錨點 {last_checkpoint}，啟動狀態無損恢復...")
    else:
        print("\n🚀 開始全新 CPT 物理隔離防作弊訓練 (From Scratch)...")

    trainer.train(resume_from_checkpoint=last_checkpoint)

    final_path = os.path.join(SAVE_DIR, "final_agiv2_cpt_1k.pth")
    torch.save(
        (
            model.state_dict()
            if not hasattr(model, "module")
            else model.module.state_dict()
        ),
        final_path,
    )
    print(f"🎉 第一階 CPT 拓撲對齊完成！最終權重已沉澱至 {final_path}")


if __name__ == "__main__":
    main()
