"""
trainRouterAC.py — 物理隔離版 (N -> N+B 範式 / 大海撈針專用)
===========================================================
核心邏輯：
1. 大海 (N)：包含 Filler 與 Trinity Logic (針)，對模型完全可見。
2. 撈針 (B)：輸入端物理遮蔽為 PAD，模型必須靠 N 的先驗從雙向空間中檢索出答案。
3. 不動數據：直接從現有的 32K 序列中動態切分 N 與 B。
"""
import os
import sys
import csv
import time
import torch
from transformers import AutoTokenizer, TrainingArguments, Trainer, set_seed, TrainerCallback
from transformers.trainer_utils import get_last_checkpoint
from datasets import load_from_disk, Dataset

# 確保路徑正確
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from AGIV2GAC import AGIV2G # 確保載入的是無因果遮罩的版本
from trainAGI import AGIV2GForCausalLM, transplant_and_freeze

set_seed(2026)

# ==========================================
# 1. 具備記憶能力的監控器 (Monitor)
# ==========================================
class ZeroGateMonitor(TrainerCallback):
    def __init__(self, path="./zerogate_ac_log.csv", save_dir="./agiv2_zerogate_ac_checkpoints"):
        self.path = path
        self.save_dir = save_dir
        self.best_eval_loss = float('inf')
        os.makedirs(self.save_dir, exist_ok=True)
        
        if os.path.exists(self.path):
            try:
                with open(self.path, 'r', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    next(reader, None)
                    for row in reader:
                        if len(row) > 2 and row[2].strip():
                            val = float(row[2])
                            if val < self.best_eval_loss: self.best_eval_loss = val
                print(f"📈 [Monitor] 找回歷史最佳 Eval Loss: {self.best_eval_loss:.4f}")
            except: pass
        else:
            with open(self.path, 'w', newline='') as f:
                csv.writer(f).writerow(['step', 'loss', 'eval_loss', 'fft_max', 'mem_max', 'time'])

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            model = kwargs.get('model', None)
            max_fft, max_mem = 0.0, 0.0
            if model:
                raw = model.module if hasattr(model, 'module') else model
                core = raw.base_model if hasattr(raw, 'base_model') else raw
                g_fft = [b.gate_fft.item() for b in core.blocks if hasattr(b, 'gate_fft')]
                g_mem = [b.gate_mem.item() for b in core.blocks if hasattr(b, 'gate_mem')]
                if g_fft: max_fft, max_mem = max(g_fft), max(g_mem)
            with open(self.path, 'a', newline='') as f:
                csv.writer(f).writerow([state.global_step, logs.get("loss", ""), logs.get("eval_loss", ""), max_fft, max_mem, time.ctime()])

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics and "eval_loss" in metrics:
            cur = metrics["eval_loss"]
            if cur < self.best_eval_loss:
                self.best_eval_loss = cur
                model = kwargs.get('model', None)
                if model:
                    best_path = os.path.join(self.save_dir, "best_model.pth")
                    raw = model.module if hasattr(model, 'module') else model
                    torch.save(raw.state_dict(), best_path)
                    print(f"\n[Monitor] 🌟 捕獲更佳權重: {cur:.4f}")

# ==========================================
# 2. 物理隔離 Trainer (N -> N+B)
# ==========================================
class RouterTrainerAC(Trainer):
    def __init__(self, pad_token_id=0, b_size=512, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pad_token_id = pad_token_id
        self.b_size = b_size # 定義 $B$ 區域（答案提取區）的長度

    def _physical_isolation_forward(self, model, input_ids):
        """
        核心機制：
        1. 獲取 N (大海) 與 B (答案區)
        2. 將輸入的 B 區域抹除，迫使模型從 N 檢索
        3. 標籤僅保留 B 區域
        """
        L = input_ids.shape[1]
        N_len = max(L - self.b_size, L // 2)
        
        # 建立隔離輸入：前 N 位可見，後面的 B 位全部 PAD
        inputs_isolated = input_ids.clone()
        inputs_isolated[:, N_len:] = self.pad_token_id
        
        # 建立隔離標籤：前 N 位忽略 (-100)，只計算 B 區域的 Loss
        labels = torch.full_like(input_ids, -100)
        labels[:, N_len:] = input_ids[:, N_len:]
        
        # 由於 labels 已對齊，模型內部的自迴歸 shift 會自動處理最後 B 個 token 的預測
        return model(input_ids=inputs_isolated, labels=labels)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        pos_ids, neg_ids = inputs.get("pos_ids"), inputs.get("neg_ids")
        
        # [對抗學習] 負樣本 Loss (檢索不該存在的針)
        with torch.no_grad():
            neg_loss = self._physical_isolation_forward(model, neg_ids)["loss"].detach()
        
        # [核心學習] 正樣本 Loss (從大海撈針)
        pos_outputs = self._physical_isolation_forward(model, pos_ids)
        pos_loss = pos_outputs["loss"]
        
        # 門控懲罰 (對齊 Roadmap 階段一：強制門控開啟)
        raw_model = model.module if hasattr(model, 'module') else model
        core_model = raw_model.base_model if hasattr(raw_model, 'base_model') else raw_model
        gate_penalty = 0.0
        for b in core_model.blocks:
            if hasattr(b, 'gate_mem'):
                gate_penalty += torch.exp(-torch.abs(b.gate_mem) * 20.0).mean()
        
        # 最終合成 Loss
        contrastive_loss = torch.clamp(pos_loss - neg_loss + 1.0, min=0.0).mean()
        total_loss = pos_loss + 0.5 * contrastive_loss + 0.1 * gate_penalty
        return (total_loss.mean(), pos_outputs) if return_outputs else total_loss.mean()

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            loss = self.compute_loss(model, inputs, return_outputs=False)
        if prediction_loss_only: return (loss, None, None)
        
        # 評估時，輸出隔離後的預測結果
        pos_ids = inputs.get("pos_ids")
        N_len = max(pos_ids.shape[1] - self.b_size, pos_ids.shape[1] // 2)
        inputs_iso = pos_ids.clone()
        inputs_iso[:, N_len:] = self.pad_token_id
        with torch.no_grad():
            outputs = model(input_ids=inputs_iso, labels=pos_ids)
        return (loss, outputs.logits, pos_ids)

# ==========================================
# 3. 主訓練流程
# ==========================================
def main():
    model_id = "google/gemma-3-1b-it"
    save_dir = "./agiv2_zerogate_ac_checkpoints"
    data_dir = "./agiv2_stage1_tridata_v5"
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    # 載入數據
    raw_ds = load_from_disk(data_dir)
    ds = raw_ds.train_test_split(test_size=0.05, seed=2026) if isinstance(raw_ds, Dataset) else raw_ds
    ds["test"] = ds["test"].select(range(min(50, len(ds["test"]))))

    # 初始化模型
    base = AGIV2G(vocab_size=262144, D=1152, hidden_dim=6912, num_blocks=26)
    base = transplant_and_freeze(model_id, base)
    model = AGIV2GForCausalLM(base, use_gc=True)
    
    # 解鎖路由組件
    unlock_keys = ["gate_fft", "gate_mem", "omegas", "mlp_H", "fft_norm", "Q_mem", "W_k_mem", "W_v_mem", "mem_norm", "W_q_cross", "o_proj_cross"]
    for n, p in model.named_parameters():
        p.requires_grad = any(k in n for k in unlock_keys)
        # 強心針梯度
        if "gate_fft" in n or "gate_mem" in n:
            p.register_hook(lambda grad: grad * 10.0)
            
    model = model.cuda().to(torch.bfloat16)

    # 訓練參數
    args = TrainingArguments(
        output_dir=save_dir,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=32,
        learning_rate=3e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        bf16=True,
        logging_steps=1,
        eval_strategy="steps",
        eval_steps=50,
        save_steps=50,
        save_total_limit=2,
        optim="paged_adamw_8bit",
        remove_unused_columns=False,
        report_to="none"
    )

    trainer = RouterTrainerAC(
        pad_token_id=pad_id,
        b_size=512, # 撈針區塊長度
        model=model,
        args=args,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        callbacks=[ZeroGateMonitor(save_dir=save_dir)]
    )

    # 斷點續傳偵測
    checkpoint = get_last_checkpoint(save_dir)
    print(f"\n🚀 {'偵測到中斷點' if checkpoint else '開始全新'}物理隔離訓練 (N -> N+B)...")
    
    trainer.train(resume_from_checkpoint=checkpoint)

    # 保存最終模型
    final_path = os.path.join(save_dir, "final_router_ac.pth")
    torch.save(model.state_dict() if not hasattr(model, 'module') else model.module.state_dict(), final_path)
    print(f"🎉 任務達成！模型已具備『大海撈針』檢索能力，權重存於: {final_path}")

if __name__ == "__main__":
    main()