import os
import sys
import torch
import gc
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
from peft import LoraConfig, get_peft_model

# 加入自己的路徑以載入先前寫好的組件 (例如 Dataset)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from AGIV2G import AGIV2G
from train import transplant_and_freeze, Packed32KDataset

# GPU 0 對齊 train.py
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["BITSANDBYTES_NOWELCOME"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

set_seed(42)


class LORAMonitor(TrainerCallback):
    def __init__(self, path="./lora_cpt_log.csv", save_dir="./lora-cpt"):
        self.path = path
        self.save_dir = save_dir
        self.best_eval_loss = float("inf")

        os.makedirs(self.save_dir, exist_ok=True)

        if not os.path.exists(self.path):
            with open(self.path, "w", newline="") as f:
                csv.writer(f).writerow(
                    [
                        "step",
                        "train_loss",
                        "eval_loss",
                        "lr",
                        "gate_fft",
                        "gate_mem",
                        "time",
                    ]
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
                        "N/A",  # LoRA 不包含 gate_fft
                        "N/A",  # LoRA 不包含 gate_mem
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
                    best_model_path = os.path.join(self.save_dir, "best_model")
                    model.save_pretrained(best_model_path)
                    print(
                        f"\n[Monitor] 🌟 Eval Loss 進步 ({old_best:.4f} -> {current_eval_loss:.4f})，已保存最佳 LoRA Adapters 至 {best_model_path}"
                    )


def main():
    model_id = "google/gemma-3-1b-it"
    save_dir = "./lora-cpt"
    os.makedirs(save_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ----------------------------------------------------
    # 1. 自動匹配 AGIV2G 的可訓練路由參數總量
    # ----------------------------------------------------
    print("🔄 正在分析原 AGIV2G 的可訓練參數總量...")
    dummy_base = AGIV2G(vocab_size=262144, D=1152, hidden_dim=6912, num_blocks=26)
    dummy_base = transplant_and_freeze(model_id, dummy_base)
    target_params = sum(p.numel() for p in dummy_base.parameters() if p.requires_grad)
    print(f"🎯 目標對標參數數量: {target_params:,}")

    del dummy_base
    gc.collect()
    torch.cuda.empty_cache()

    # ----------------------------------------------------
    # 2. 載入原始模型並計算 LoRA 最佳 R 值
    # ----------------------------------------------------
    print(f"🔄 正在載入原始模型 {model_id}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="cuda"
    )

    # 尋找指定 Linear 模組來套用 LoRA
    target_modules = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]

    params_per_r = 0
    for name, module in model.named_modules():
        if any(t in name for t in target_modules) and hasattr(module, "weight"):
            idx_out, idx_in = module.weight.shape
            params_per_r += idx_in + idx_out

    import math

    optimal_r = max(1, math.ceil(target_params / params_per_r))
    print(f"🤖 經純數學推算，每一單位的 r 將帶來 {params_per_r:,} 參數")
    print(f"🤩 最佳 LoRA Rank (r) 為 {optimal_r} 以完美對標 AGIV2G 的負載")

    # 套用 LoRA
    config = LoraConfig(
        r=optimal_r,
        lora_alpha=optimal_r * 2,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)

    final_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"✅ LoRA 初始化完成，實際訓練參數: {final_params:,} (與目標差距: {final_params - target_params:,})"
    )

    # ----------------------------------------------------
    # 3. 準備資料與開始訓練
    # ----------------------------------------------------
    train_ds = Packed32KDataset(tokenizer, max_length=32768, is_val=False)
    val_ds = Packed32KDataset(tokenizer, max_length=32768, is_val=True, max_samples=20)

    args = TrainingArguments(
        output_dir=save_dir,
        max_steps=5000,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,  # 防止 Eval OOM，這點與最佳設定對齊
        gradient_accumulation_steps=8,
        learning_rate=1e-4,
        warmup_steps=250,
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
        gradient_checkpointing=True,  # 開啟官方模型的梯度檢查點，這是防止標準模型 OOM 的關鍵
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        callbacks=[LORAMonitor(save_dir=save_dir)],
    )

    # 對齊 Resume From Checkpoint 機制
    last_checkpoint = get_last_checkpoint(save_dir)
    if last_checkpoint is not None:
        print(f"\n🚀 偵測到極限中斷點 {last_checkpoint}，啟動 LoRA 狀態恢復...")
    else:
        print("\n🚀 開始 LORA 32K CPT 全新訓練 (CUDA:0)...")

    trainer.train(resume_from_checkpoint=last_checkpoint)

    # 儲存最終 LoRA adapters
    model.save_pretrained(os.path.join(save_dir, "final"))
    print("🎉 LORA 訓練完成，Adapters 已儲存！")


if __name__ == "__main__":
    main()
