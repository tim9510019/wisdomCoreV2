# trainCPTAC_1080Ti.py — AGIV2 1080 Ti 原生訓練節奏 (升級 BF16 + 主動式散熱 + 10倍速等效 TQDM 完美列印版)
import os
import sys
import csv
import time
import torch
import random
from tqdm.auto import tqdm
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    set_seed,
    TrainerCallback,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.trainer_callback import PrinterCallback
from datasets import load_dataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 1080 Ti 環境優化 (全數保留)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["BITSANDBYTES_NOWELCOME"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_SKIP_CHECK_TORCH_LOAD_SAFE"] = "True"

from AGIV2GACT import AGIV2G
from utils import AGIV2GForCausalLMT

# ==========================================
# [ 核心配置區 ]
# ==========================================
MODEL_ID = "google/gemma-3-1b-it"
DATASET_DIR = "./agiv2_stage1_1K"
SAVE_DIR = "./agiv2_cpt_1080Ti_10X_checkpoints"
LOG_PATH = "./agiv2_cpt_1080Ti_10X_log.csv"
BEST_ROUTER_32K_PATH = "./agiv2_zerogate_ac_checkpoints_32KST/best_model.pth"
RANDOM_SEED = 2026

D_MODEL = 768
H_DIM = 3072
N_BLOCKS = 12
CHUNK = 256

LISA_N_ACTIVE = 2
LISA_INTERVAL = 100

MAX_STEPS = 1678120
WARMUP_STEPS = 500
EVAL_STEPS = 100
SAVE_STEPS = 100
SAVE_TOTAL_LIMIT = 2
LOGGING_STEPS = 1
BATCH_SIZE_PER_DEVICE = 2
GRAD_ACCUMULATION_STEPS = 16
LEARNING_RATE = 2e-4

set_seed(RANDOM_SEED)


# ==========================================
# [ 硬體保護：主動式散熱控制器 ]
# ==========================================
class ThermalControlCallback(TrainerCallback):
    def __init__(self, delay_seconds=0.5):
        self.delay_seconds = delay_seconds

    def on_step_end(self, args, state, control, **kwargs):
        time.sleep(self.delay_seconds)


# ==========================================
# [ 隱身型 TQDM 控制器 (無廢話、不換行、完美還原 log) ]
# ==========================================
class StealthTqdmCallback(TrainerCallback):
    def __init__(self):
        self.pbar = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.pbar = tqdm(
            total=MAX_STEPS, desc="Iteration", dynamic_ncols=True, leave=True
        )

    def on_step_end(self, args, state, control, **kwargs):
        if self.pbar is not None:
            self.pbar.update(10)  # 物理 1 步，視覺跳 10 步

    def on_log(self, args, state, control, logs=None, **kwargs):
        if self.pbar is not None and logs:
            if "loss" in logs:
                self.pbar.set_postfix(loss=f"{logs['loss']:.4f}")

            # [修正點] 過濾並格式化字典，重現原生 HuggingFace 的列印行為
            display_logs = {}
            for k, v in logs.items():
                if k in [
                    "loss",
                    "grad_norm",
                    "learning_rate",
                    "epoch",
                    "g_loc",
                    "g_mem",
                    "g_fft",
                ]:
                    if isinstance(v, float):
                        if k == "learning_rate":
                            display_logs[k] = f"{v:.4e}" if v < 0.001 else f"{v:.4f}"
                        else:
                            display_logs[k] = str(round(v, 4))
                    else:
                        display_logs[k] = str(v)

            # 使用 tqdm.write 會把文字「推」到進度條上方，絕對不會讓進度條斷掉
            if display_logs and "loss" in display_logs:
                tqdm.write(str(display_logs))

    def on_train_end(self, args, state, control, **kwargs):
        if self.pbar is not None:
            self.pbar.close()


# ==========================================
# [ 監控器 (背景還原真實等效步數) ]
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
                        f"📈 [Monitor] 當前最佳 Eval Loss 基準: {self.best_eval_loss:.4f}"
                    )
            except Exception as e:
                pass
        else:
            with open(self.path, "w", newline="") as f:
                csv.writer(f).writerow(
                    ["step", "loss", "eval_loss", "g_loc", "g_mem", "g_fft", "time"]
                )

    def on_step_begin(self, args, state, control, **kwargs):
        model = kwargs.get("model", None)
        if model is not None:
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
            g_loc_avg, g_mem_avg, g_fft_avg = 0.0, 0.0, 0.0
            blocks_count = 0
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
                        g_fft_avg += b.avg_g_fft.item()
                        blocks_count += 1
            if blocks_count > 0:
                g_loc_avg /= blocks_count
                g_mem_avg /= blocks_count
                g_fft_avg /= blocks_count

            # 這一步確保後續的 StealthTqdmCallback 可以抓到這三個自訂參數
            logs["g_loc"] = round(g_loc_avg, 4)
            logs["g_mem"] = round(g_mem_avg, 4)
            logs["g_fft"] = round(g_fft_avg, 4)

            with open(self.path, "a", newline="") as f:
                csv.writer(f).writerow(
                    [
                        state.global_step * 10,
                        logs.get("loss", ""),
                        logs.get("eval_loss", ""),
                        f"{g_loc_avg:.4f}",
                        f"{g_mem_avg:.4f}",
                        f"{g_fft_avg:.4f}",
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
                    best_model_path = os.path.join(
                        self.save_dir, "best_cpt_model.safetensors"
                    )
                    raw_model = model.module if hasattr(model, "module") else model
                    from safetensors.torch import save_model

                    save_model(raw_model, best_model_path)
                    tqdm.write(
                        f"[Monitor] 🌟 收斂突破 ({old_best:.4f} -> {current_eval_loss:.4f})，儲存至 {best_model_path}"
                    )


# ==========================================
# [ LISA (Layerwise Importance Sampling) 控制器 ]
# ==========================================
class LISATrainingCallback(TrainerCallback):
    def __init__(self, n_blocks=12, n_active=2, interval=100):
        self.n_blocks = n_blocks
        self.n_active = n_active
        self.interval = interval
        self.last_step = -1

    def on_step_begin(self, args, state, control, **kwargs):
        step = state.global_step
        if step % self.interval == 0 and step != self.last_step:
            self.last_step = step
            model = kwargs.get("model", None)
            if model is not None:
                self.apply_lisa_sampling(model)

    def apply_lisa_sampling(self, model):
        raw_model = model.module if hasattr(model, "module") else model
        core_model = (
            raw_model.base_model if hasattr(raw_model, "base_model") else raw_model
        )

        for param in model.parameters():
            param.requires_grad = False
        if hasattr(core_model, "embed"):
            for param in core_model.embed.parameters():
                param.requires_grad = True
        if hasattr(raw_model, "lm_head"):
            for param in raw_model.lm_head.parameters():
                param.requires_grad = True

        window_index = self.last_step // self.interval
        local_seed = RANDOM_SEED + window_index
        rng = random.Random(local_seed)

        active_indices = rng.sample(list(range(self.n_blocks)), self.n_active)
        active_indices.sort()
        active_str = ", ".join([str(i) for i in active_indices])

        tqdm.write(
            f"💉 [LISA] 執行隨機層採樣 (種子:{local_seed})：選中 Blocks [{active_str}]"
        )

        for i in active_indices:
            for param in core_model.blocks[i].parameters():
                param.requires_grad = True


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
# [ 核心重構：STEP 級別邊界過濾器 (隱身模式) ]
# ==========================================
class FilteredDataLoaderWrapper:
    def __init__(self, dataloader, model, trainer, group_size_steps, accum_steps):
        self.dataloader = dataloader
        self.model = model
        self.trainer = trainer
        self.group_size_steps = group_size_steps
        self.accum_steps = accum_steps

    def __len__(self):
        return max(1, len(self.dataloader) // 10)

    def __iter__(self):
        iterator = iter(self.dataloader)

        while True:
            step_chunks = []

            for _ in range(self.group_size_steps):
                current_step_batches = []
                for _ in range(self.accum_steps):
                    try:
                        batch = next(iterator)
                        current_step_batches.append(batch)
                    except StopIteration:
                        break
                if current_step_batches:
                    step_chunks.append(current_step_batches)
                else:
                    break

            if not step_chunks:
                break

            self.model.eval()
            step_losses = []
            with torch.no_grad():
                for chunk in step_chunks:
                    chunk_loss_sum = 0.0
                    for batch in chunk:
                        prepared_batch = self.trainer._prepare_inputs(batch)
                        with self.trainer.compute_loss_context_manager():
                            loss_output = self.trainer.compute_loss(
                                self.model, prepared_batch, return_outputs=False
                            )
                            loss = (
                                loss_output[0]
                                if isinstance(loss_output, tuple)
                                else loss_output
                            )
                        chunk_loss_sum += loss.item()
                        del prepared_batch
                    step_losses.append(chunk_loss_sum / len(chunk))

            self.model.train()

            n = len(step_chunks)
            if n < 10:
                selected_chunks = step_chunks
            else:
                sorted_indices = sorted(range(n), key=lambda i: step_losses[i])
                start_idx = int(n * 0.55)
                end_idx = max(start_idx + 1, int(n * 0.65))
                selected_indices = sorted_indices[start_idx:end_idx]
                selected_chunks = [step_chunks[i] for i in selected_indices]

            for chunk in selected_chunks:
                for b in chunk:
                    yield b


class AcceleratedTrainer(Trainer):
    def get_train_dataloader(self):
        dataloader = super().get_train_dataloader()
        return FilteredDataLoaderWrapper(
            dataloader,
            self.model,
            self,
            group_size_steps=SAVE_STEPS,
            accum_steps=GRAD_ACCUMULATION_STEPS,
        )


# ==========================================
# [ 核心引擎 ]
# ==========================================
def main():
    print(f"\n🚀 啟動 AGIV2 優化矩陣 (D={D_MODEL}, Blocks={N_BLOCKS})...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    parquet_file = os.path.join(DATASET_DIR, "agiv2_stage1_N_B.parquet")
    if not os.path.exists(parquet_file):
        print(f"❌ 找不到 Parquet 資料: {parquet_file}")
        return

    dataset_dict = load_dataset("parquet", data_files=parquet_file)

    dataset = dataset_dict["train"].train_test_split(test_size=50, seed=RANDOM_SEED)
    train_ds, eval_ds = dataset["train"], dataset["test"]
    print(
        f"✅ 資料映射與絕對隔離完成！訓練集: {len(train_ds):,} 筆 | 獨立驗證集: {len(eval_ds)} 筆"
    )

    base = AGIV2G(
        vocab_size=tokenizer.vocab_size,
        D=D_MODEL,
        C=CHUNK,
        hidden_dim=H_DIM,
        num_blocks=N_BLOCKS,
    )
    model = AGIV2GForCausalLMT(base, use_gc=True)

    if os.path.exists(BEST_ROUTER_32K_PATH):
        if BEST_ROUTER_32K_PATH.endswith(".safetensors"):
            from safetensors.torch import load_model

            load_model(model, BEST_ROUTER_32K_PATH, strict=False)
            print("✅ Safetensors 路由權重載入完成。")

    model = model.cuda()

    args = TrainingArguments(
        output_dir=SAVE_DIR,
        max_steps=max(1, MAX_STEPS // 10),
        per_device_train_batch_size=BATCH_SIZE_PER_DEVICE,
        per_device_eval_batch_size=BATCH_SIZE_PER_DEVICE,
        gradient_accumulation_steps=GRAD_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_steps=max(1, WARMUP_STEPS // 10),
        bf16=True,
        logging_steps=LOGGING_STEPS,
        eval_strategy="steps",
        eval_steps=max(1, EVAL_STEPS // 10),
        save_steps=max(1, SAVE_STEPS // 10),
        save_total_limit=SAVE_TOTAL_LIMIT,
        optim="paged_adamw_8bit",
        weight_decay=0.1,
        remove_unused_columns=False,
        report_to="none",
        disable_tqdm=True,
    )

    trainer = AcceleratedTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=CPTDataCollator(pad_id),
        callbacks=[
            # [關鍵修正順序] Monitor 必須在前面，才能先算完 g_loc 塞進 logs 給 TQDM 印出來
            QuantumCPTMonitor(path=LOG_PATH, save_dir=SAVE_DIR),
            StealthTqdmCallback(),
            LISATrainingCallback(
                n_blocks=N_BLOCKS,
                n_active=LISA_N_ACTIVE,
                interval=max(1, LISA_INTERVAL // 10),
            ),
            ThermalControlCallback(delay_seconds=1.5),
        ],
    )

    trainer.remove_callback(PrinterCallback)

    last_checkpoint = get_last_checkpoint(SAVE_DIR)
    if last_checkpoint:
        weight_path = os.path.join(last_checkpoint, "model.safetensors")
        if not os.path.exists(weight_path):
            weight_path = os.path.join(last_checkpoint, "pytorch_model.bin")
        if os.path.exists(weight_path):
            if weight_path.endswith(".safetensors"):
                from safetensors.torch import load_model

                load_model(model, weight_path, strict=False)
            else:
                model.load_state_dict(
                    torch.load(weight_path, map_location="cpu"), strict=False
                )
            print("✅ 權重恢復完成。")

    print("\n🔧 啟動 LISA 演算法 (外科手術級隨機層更新模式)...")
    trainer.train(resume_from_checkpoint=None)

    final_path = os.path.join(SAVE_DIR, "final_agiv2_cpt_1080ti.safetensors")
    from safetensors.torch import save_model

    save_model(model if not hasattr(model, "module") else model.module, final_path)


if __name__ == "__main__":
    main()
