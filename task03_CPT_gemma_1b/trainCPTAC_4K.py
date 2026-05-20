"""
trainCPTAC_4K.py — AGIV2 持續預訓練終極引擎 (Stage 2: 4K 視界擴展版)
===========================================================
第一性原理實踐：
1. 跨階拓撲繼承：精準載入 1K 階段 Best Model，在已覺醒的局部路由基礎上，強制擴展 4K 全局共振。
2. Smart Token Init：將原生空白符號 <unused0> 的權重初始化為換行與 EOS 的平均疊加態，消除高頻雜訊。
3. 退火震盪容忍：4K 序列下的 Loss 震盪是尋找全局最優盆地的必然物理現象，監控器將專注於 Gate 極化程度。
4. 物理隔離直讀：DataCollator 釋放真實訊號，n_split_index 穿透底層執行 NAR 隔離，表層執行 AR 收斂。
"""

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

# 確保外部依賴路徑暢通
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 算力池綁定
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["BITSANDBYTES_NOWELCOME"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from AGIV2GAC import AGIV2G
from utils import AGIV2GForCausalLM, transplant_and_freeze

# ==========================================
# [ 全局配置區 ] 能量分佈與物理參數控制台
# ==========================================
RANDOM_SEED = 2026
set_seed(RANDOM_SEED)

MODEL_ID = "google/gemma-3-1b-it"
DOC_SEP_TOKEN = "<unused0>"

# 📂 資料與儲存拓撲
DATASET_DIR = "./agiv2_stage2_4K"
SAVE_DIR = "./agiv2_cpt_checkpoints_4K"
LOG_PATH = "./agiv2_cpt_4k_log.csv"

# 🌟 跨階段拓撲繼承：指向 1K 階段的最佳權重
BEST_MODEL_1K_PATH = "./agiv2_cpt_checkpoints_1K/best_cpt_model.pth"

# ⚙️ 4K 階段訓練超參數 (Stage 2: Plateau)
MAX_STEPS = 85937
WARMUP_STEPS = 500
EVAL_STEPS = 100
SAVE_STEPS = 100
SAVE_TOTAL_LIMIT = 3
LOGGING_STEPS = 1

# ⚠️ 記憶體防護：4K 序列消耗極大，Batch Size 下調，Grad Accumulation 等比例放大
BATCH_SIZE_PER_DEVICE = 1
GRAD_ACCUMULATION_STEPS = 32
LEARNING_RATE = 1e-4  # 進入 CPT 退火高原期


# ==========================================
# [ 監控維度 ] 退火與量子震盪觀測器
# ==========================================
class QuantumCPTMonitor(TrainerCallback):
    def __init__(self, path=LOG_PATH, save_dir=SAVE_DIR):
        self.path = path
        self.save_dir = save_dir
        self.best_loss = float("inf")
        os.makedirs(self.save_dir, exist_ok=True)

        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    reader = csv.reader(f)
                    next(reader, None)
                    for row in reader:
                        if len(row) > 1 and row[1].strip():
                            try:
                                val = float(row[1])
                                if val < self.best_loss:
                                    self.best_loss = val
                            except ValueError:
                                pass
                if self.best_loss != float("inf"):
                    print(
                        f"\n📈 [Monitor] 時間線同步成功。當前最佳 4K Training Loss 基準: {self.best_loss:.4f}"
                    )
            except Exception as e:
                print(f"\n⚠️ [Monitor] 讀取歷史紀錄失敗: {e}")
        else:
            with open(self.path, "w", newline="") as f:
                csv.writer(f).writerow(["step", "loss", "fft_max", "mem_max", "time"])

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

            current_loss = logs.get("loss", None)
            with open(self.path, "a", newline="") as f:
                csv.writer(f).writerow(
                    [state.global_step, current_loss, max_fft, max_mem, time.ctime()]
                )

            # 4K 退火震盪防護：只有在 Loss 創下新低且 Gate 表現活躍時，才判定為結構極化收斂
            if current_loss and current_loss < self.best_loss:
                self.best_loss = current_loss
                if model is not None and state.global_step % (SAVE_STEPS // 2) == 0:
                    best_model_path = os.path.join(
                        self.save_dir, "best_cpt_model_4k.pth"
                    )
                    raw_model = model.module if hasattr(model, "module") else model
                    torch.save(raw_model.state_dict(), best_model_path)
                    print(
                        f"\n[Monitor] 🌌 4K 結構極化收斂 ({current_loss:.4f})，實體權重已錨定。"
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

            # 🌟 核心修改：非對稱解耦 (Asymmetric Decoupling)
            # 釋放真實訊號，不抹除 B 區塊，保留完整 [N, B] 讓 Local SDPA 執行 AR 接龍
            # M_global 的 NAR 物理隔斷，將由穿透下去的 n_split_index 在模型內部執行
            pad_len = max_len - len(seq)
            padded_input = seq + [self.pad_token_id] * pad_len
            batch_input_ids.append(padded_input)

            # 🌟 預測目標不變：前 N 個不計算 Loss (-100)，只針對 B 區塊計算 AR 預測的 Loss
            labels = [-100] * n_split + seq[n_split:] + [-100] * pad_len
            batch_labels.append(labels)

            # 🌟 收集切分點，準備穿透至底層
            batch_n_splits.append(n_split)

        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
            "n_split_index": torch.tensor(
                batch_n_splits, dtype=torch.long
            ),  # 🌟 穿透參數，傳遞給底層進行物理截斷
        }


# ==========================================
# [ 核心引擎 ] 主執行緒與外科手術解凍
# ==========================================
def main():
    print("\n🚀 啟動 AGIV2 CPT 第二階 (4K) 訓練矩陣...")

    # 1. 初始化 Tokenizer 與原生符號定錨
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    if DOC_SEP_TOKEN not in tokenizer.get_vocab():
        print(f"⚠️ 警告：詞表中未發現 {DOC_SEP_TOKEN}，這違反了 4K 原生符號徵用協議！")
        sys.exit(1)

    doc_sep_id = tokenizer.convert_tokens_to_ids(DOC_SEP_TOKEN)
    newline_id = tokenizer.convert_tokens_to_ids("\n")
    eos_id = tokenizer.eos_token_id

    # 2. 載入超光速 4K 資料庫
    parquet_file = os.path.join(DATASET_DIR, "agiv2_stage2_N_B_4K.parquet")
    if not os.path.exists(parquet_file):
        raise FileNotFoundError(
            f"找不到 4K 實體矩陣 {parquet_file}，請先執行 cpt4k.py。"
        )

    print(f"📦 成功鎖定 4K 實體磁區: {parquet_file}")
    dataset_dict = load_dataset("parquet", data_files=parquet_file)

    dataset = dataset_dict["train"].train_test_split(test_size=50, seed=RANDOM_SEED)
    train_ds = dataset["train"]
    eval_ds = dataset["test"]
    print(
        f"✅ 資料隔離完成！訓練集: {len(train_ds):,} 筆 | 獨立驗證集: {len(eval_ds)} 筆"
    )

    # 3. 模型拓撲對齊
    base = AGIV2G(vocab_size=262144, D=1152, C=256, hidden_dim=6912, num_blocks=26)
    base = transplant_and_freeze(MODEL_ID, base)
    model = AGIV2GForCausalLM(base, use_gc=True)

    # 🌟 [ 跨階段拓撲繼承 ] 載入 1K 最佳狀態
    if os.path.exists(BEST_MODEL_1K_PATH):
        print(f"\n🔄 尋獲 1K 階段最佳矩陣，啟動時間線繼承: {BEST_MODEL_1K_PATH}")
        state_dict = torch.load(BEST_MODEL_1K_PATH, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
        print("✅ 1K 局部高精度路由能力已成功灌注。")
    else:
        print(f"\n⚠️ 找不到 1K 權重 {BEST_MODEL_1K_PATH}，宇宙將從零開始坍縮！")

    # 🌟 [ Smart Token Init ] 平均池化初始化
    print("\n🔧 執行 Smart Token Init (實體注能)...")
    with torch.no_grad():
        embeddings = model.get_input_embeddings().weight.data
        embeddings[doc_sep_id] = (embeddings[newline_id] + embeddings[eos_id]) / 2.0
    print(f"✅ {DOC_SEP_TOKEN} 已被賦予換行與 EOS 的疊加態語義。")

    # 4. 外科手術解凍矩陣
    # 🌟 修改點：增量解凍法 (對齊 1K 的修正，不再漏掉任何新增參數)
    norm_keys = ["norm", "input_layernorm", "post_attention_layernorm"]
    frozen_keys = ["lm_head", "o_proj"]

    print("\n🔧 啟動外科手術解凍程序 (CPT 4K Stage):")
    # 繼承 transplant_and_freeze 的狀態 (AGIV2新層皆為 True，Gemma 舊層為 False)
    for name, param in model.named_parameters():
        # 1. 額外解凍 Gemma 的 Norm 參數，以適應真實語義分佈
        if any(k in name for k in norm_keys):
            param.requires_grad = True

        # 2. 保險防護：確保原始的輸出與映射層絕對不被動到 (防範意外匹配)
        if any(k in name for k in frozen_keys) and "o_proj_cross" not in name:
            param.requires_grad = False

        # 3. 註冊動態鎖相 (Dynamic Phase-Locking) 的梯度放大器
        if ("gate_fft" in name or "gate_mem" in name) and param.requires_grad:
            param.register_hook(lambda grad: grad * 10.0)  # 保持梯度強心針

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✅ 解凍完成。可訓練參數總量: {trainable_params:,}")

    model = model.cuda().to(torch.bfloat16)

    # 5. 訓練能量配置
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

    # 6. 啟動框架
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
        print(f"\n🚀 偵測到 4K 時間錨點 {last_checkpoint}，啟動狀態無損恢復...")
    else:
        print("\n🚀 開始全新 4K CPT 物理隔離訓練 (N -> N+B)...")

    trainer.train(resume_from_checkpoint=last_checkpoint)

    # 7. 最終坍縮
    final_path = os.path.join(SAVE_DIR, "final_agiv2_cpt_4k.pth")
    torch.save(
        (
            model.state_dict()
            if not hasattr(model, "module")
            else model.module.state_dict()
        ),
        final_path,
    )
    print(f"🎉 第二階 CPT 拓撲對齊完成！4K 全局共振權重已沉澱至 {final_path}")


if __name__ == "__main__":
    main()
