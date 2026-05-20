import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import csv
import time
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from datasets import load_dataset
from torch.utils.data import Dataset

from AGIV2G import AGIV2G

# 直接匯入你在 train.py 寫好的神級外殼與邏輯
from train import AGIV2GForCausalLM, transplant_and_freeze

# ==========================================
# 0. 環境設定
# ==========================================
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["BITSANDBYTES_NOWELCOME"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
set_seed(42)


# ==========================================
# 1. 業界標準 SFT 資料集 (具備 Loss Masking 機制)
# ==========================================
class SFTDataset(Dataset):
    def __init__(self, tokenizer, split="train", max_length=32768, max_samples=None):
        """
        載入 Cerebras 的 32K 合成巨兵資料庫 (Synth-Long-SFT32K)
        全覆蓋 32K 長度，針對性壓榨 NIAH 極限！
        """
        print(f"🔄 正在載入 32K 極限考卷：Synth-Long-SFT32K ({split})...")
        # 直接拿最暴力的 32K 考卷 split
        data = load_dataset(
            "cerebras/Synth-Long-SFT32K", split="train_narrativeqa_aug_32k"
        )

        if max_samples:
            data = data.select(range(min(max_samples, len(data))))

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []

        print("⚙️ 正在執行 32K 級的 Loss Masking 標註處理...")
        for ex in data:
            conversations = ex.get("conversations", [])
            # 確保對話結構存在
            if len(conversations) < 2:
                continue

            # Cerebras 常用的 key 是 value，我們做一下防呆
            user_msg = conversations[0].get(
                "value", conversations[0].get("content", "")
            )
            model_msg = conversations[1].get(
                "value", conversations[1].get("content", "")
            )

            # 略過空值
            if not user_msg or not model_msg:
                continue

            # 使用 Tokenizer 產生 User 提問的長度 (此部分不計算梯度)
            user_chat = [{"role": "user", "content": user_msg}]
            user_prompt = tokenizer.apply_chat_template(
                user_chat, tokenize=False, add_generation_prompt=True
            )
            user_ids = tokenizer(user_prompt, add_special_tokens=False).input_ids

            # 使用 Tokenizer 產生完整的一輪對話
            full_chat = [
                {"role": "user", "content": user_msg},
                {"role": "model", "content": model_msg},
            ]
            full_prompt = tokenizer.apply_chat_template(
                full_chat, tokenize=False, add_generation_prompt=False
            )
            # 確保結尾有 EOS
            if not full_prompt.endswith(tokenizer.eos_token):
                full_prompt += tokenizer.eos_token

            input_ids = tokenizer(full_prompt, add_special_tokens=False).input_ids

            # ✂️ 長度截斷保護
            if len(input_ids) > self.max_length:
                input_ids = input_ids[: self.max_length]
                input_ids[-1] = tokenizer.eos_token_id  # 強制補上結尾符號

            labels = list(input_ids)

            # 🛡️ 核心：把屬於 User 提問區段的 Labels 全部設定為 -100！
            # 這樣 PyTorch 的 CrossEntropyLoss 就會完全無視這段的前向預測，專心對齊 Model 的回答式子
            mask_len = min(len(user_ids), len(labels))
            for i in range(mask_len):
                labels[i] = -100

            self.examples.append(
                {
                    "input_ids": torch.tensor(input_ids, dtype=torch.long),
                    "labels": torch.tensor(labels, dtype=torch.long),
                }
            )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


# Datacollator 用來把長短不一的句子補齊 (Padding)
def sft_collate_fn(batch):
    input_ids = [item["input_ids"] for item in batch]
    labels = [item["labels"] for item in batch]

    # 找出 batch 中最長的長度
    max_len = max(len(ids) for ids in input_ids)

    padded_inputs = []
    padded_labels = []

    for inp, lbl in zip(input_ids, labels):
        pad_len = max_len - len(inp)
        # 用 0 做 input padding，用 -100 做 label padding (忽略 loss)
        padded_inputs.append(torch.cat([inp, torch.zeros(pad_len, dtype=torch.long)]))
        padded_labels.append(
            torch.cat([lbl, torch.full((pad_len,), -100, dtype=torch.long)])
        )

    return {
        "input_ids": torch.stack(padded_inputs),
        "labels": torch.stack(padded_labels),
    }


# ==========================================
# 2. 監控面版 (追蹤 SFT 的損失率)
# ==========================================
class SFTMonitor(TrainerCallback):
    def __init__(self, path="./sft_log.csv", save_dir="./agiv2-sft"):
        self.path = path
        self.save_dir = save_dir
        self.best_eval_loss = float("inf")
        os.makedirs(self.save_dir, exist_ok=True)
        if not os.path.exists(self.path):
            with open(self.path, "w", newline="") as f:
                csv.writer(f).writerow(
                    ["step", "train_loss", "eval_loss", "lr", "time"]
                )

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            with open(self.path, "a", newline="") as f:
                csv.writer(f).writerow(
                    [
                        state.global_step,
                        logs.get("loss", ""),
                        logs.get("eval_loss", ""),
                        logs.get("learning_rate", ""),
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
                    best_model_path = os.path.join(self.save_dir, "best_sft_model.pth")
                    torch.save(model.state_dict(), best_model_path)
                    print(
                        f"\n[Monitor] 🌟 SFT Eval Loss 進步 ({old_best:.4f} -> {current_eval_loss:.4f})，已淬鍊 SFT 權重至 {best_model_path}"
                    )


# ==========================================
# 3. 主流程
# ==========================================
def main():
    model_id = "google/gemma-3-1b-it"
    cpt_save_dir = "./agiv2-cpt"
    sft_save_dir = "./agiv2-sft"
    os.makedirs(sft_save_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("🔄 正在建立 AGIV2G 架構...")
    base = AGIV2G(vocab_size=262144, D=1152, hidden_dim=6912, num_blocks=26)

    # 決定從前一個 CPT 階段的最強存檔繼承，還是重新從零開始
    cpt_best_path = os.path.join(cpt_save_dir, "best_model.pth")
    cpt_final_path = os.path.join(cpt_save_dir, "final.pth")

    model = AGIV2GForCausalLM(base).to(torch.bfloat16).cuda()

    loaded = False
    for path in [cpt_best_path, cpt_final_path]:
        if os.path.exists(path):
            print(f"📦 偵測到 CPT 階段遺產！正在繼承權重 {path}，進入 SFT 覺醒模式！")
            model.load_state_dict(torch.load(path, map_location="cuda"), strict=False)
            loaded = True
            break

    if not loaded:
        print("⚠️ 警告: 找不到 CPT 訓練的紀錄，將退化回載入原始官方權重來進行 SFT...")
        base = transplant_and_freeze(model_id, base)
        model = AGIV2GForCausalLM(base).to(torch.bfloat16).cuda()

    # 準備 32K 的終極 SFT 資料
    # 這裡我們只取少量筆數即可，因為長度高達 32K，計算極度耗時且容易過擬合
    # 你可以依照算力隨時拉高 max_samples
    train_ds = SFTDataset(tokenizer, split="train", max_length=32768, max_samples=2000)
    val_ds = SFTDataset(tokenizer, split="test", max_length=32768, max_samples=20)

    # SFT 專屬參數
    args = TrainingArguments(
        output_dir=sft_save_dir,
        max_steps=1000,  # 降低訓練步數，因為資料每一筆都超長 (32K)
        per_device_train_batch_size=1,  # 32K 長度極其龐大，強制退回 BS=1
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-5,
        warmup_steps=50,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        bf16=True,
        logging_steps=1,
        save_steps=1,
        save_total_limit=2,
        eval_strategy="steps",
        eval_steps=100,
        optim="paged_adamw_8bit",
        report_to="none",
        gradient_checkpointing=False,  # 關閉，因為 AGIV2 客製化架構中已經在 forward() 內文強制自帶了 checkpoint 保護！
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=sft_collate_fn,
        callbacks=[SFTMonitor(save_dir=sft_save_dir)],
    )

    last_checkpoint = get_last_checkpoint(sft_save_dir)
    if last_checkpoint is not None:
        print(f"\n🚀 偵測到 SFT 階段中斷點 {last_checkpoint}，啟動狀態恢復...")
    else:
        print("\n🚀 開始 SFT 對齊訓練 (Supervised Fine-Tuning) (CUDA:0)...")

    trainer.train(resume_from_checkpoint=last_checkpoint)

    torch.save(model.state_dict(), os.path.join(sft_save_dir, "sft_final.pth"))
    print("🎉 SFT 人類偏好微調階段完成！")


if __name__ == "__main__":
    main()
