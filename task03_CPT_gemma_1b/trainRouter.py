"""
trainRouter_ZeroGate.py — 內源路由 (ZeroGate) 啟動訓練腳本
===========================================================
透過梯度強制推開 gate_fft 與 gate_mem，訓練網路自身的全局路由能力。
融合歷史回溯、動態 Eval Loss 淬鍊與極限斷點恢復機制。
"""
import os
import sys
import csv
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ==========================================
# 0. SOTA 環境配置
# ==========================================
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["BITSANDBYTES_NOWELCOME"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from transformers import AutoTokenizer, TrainingArguments, Trainer, set_seed, TrainerCallback
from transformers.trainer_utils import get_last_checkpoint
from datasets import load_from_disk
from AGIV2G import AGIV2G
from trainAGI import AGIV2GForCausalLM, transplant_and_freeze

set_seed(42)

# ==========================================
# 1. 具備歷史回溯能力的動態監控 (ZeroGate 版)
# ==========================================
class ZeroGateMonitor(TrainerCallback):
    def __init__(self, path="./zerogate_32k_log.csv", save_dir="./agiv2_zerogate_checkpoints"):
        self.path = path
        self.save_dir = save_dir
        self.best_eval_loss = float('inf')
        
        os.makedirs(self.save_dir, exist_ok=True)
        
        if os.path.exists(self.path):
            try:
                with open(self.path, 'r', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    header = next(reader, None)
                    if header:
                        for row in reader:
                            if len(row) > 2 and row[2].strip():
                                try:
                                    val = float(row[2])
                                    if val < self.best_eval_loss:
                                        self.best_eval_loss = val
                                except ValueError:
                                    pass
                if self.best_eval_loss != float('inf'):
                    print(f"\n📈 [Monitor] 成功讀取歷史紀錄！當前最佳 Eval Loss 基準線為: {self.best_eval_loss:.4f}")
            except Exception as e:
                print(f"\n⚠️ [Monitor] 讀取歷史紀錄失敗: {e}，將重置基準線。")
        else:
            with open(self.path, 'w', newline='') as f:
                csv.writer(f).writerow(['step', 'train_loss', 'eval_loss', 'lr', 'gate_fft_avg', 'gate_mem_avg', 'gate_fft_max', 'gate_mem_max', 'time'])

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            model = kwargs.get('model', None)
            avg_fft, avg_mem, max_fft, max_mem = 0.0, 0.0, 0.0, 0.0
            
            if model is not None:
                core_model = model.base_model if hasattr(model, 'base_model') else model
                g_fft, g_mem = [], []
                
                for block in core_model.blocks:
                    if hasattr(block, 'gate_fft'):
                        g_fft.append(block.gate_fft.item())
                        g_mem.append(block.gate_mem.item())
                
                if g_fft:
                    avg_fft = sum(g_fft) / len(g_fft)
                    avg_mem = sum(g_mem) / len(g_mem)
                    max_fft = max(g_fft)
                    max_mem = max(g_mem)
                    
                    logs["fft_avg"] = round(avg_fft, 6)
                    logs["fft_max"] = round(max_fft, 6)
                    logs["mem_avg"] = round(avg_mem, 6)
                    logs["mem_max"] = round(max_mem, 6)

            with open(self.path, 'a', newline='') as f:
                csv.writer(f).writerow([
                    state.global_step, 
                    logs.get("loss", ""), 
                    logs.get("eval_loss", ""), 
                    logs.get("learning_rate", ""), 
                    f"{avg_fft:.6f}" if g_fft else "",
                    f"{avg_mem:.6f}" if g_fft else "",
                    f"{max_fft:.6f}" if g_fft else "",
                    f"{max_mem:.6f}" if g_fft else "",
                    time.ctime()
                ])

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics and "eval_loss" in metrics:
            current_eval_loss = metrics["eval_loss"]
            if current_eval_loss < self.best_eval_loss:
                old_best = self.best_eval_loss
                self.best_eval_loss = current_eval_loss
                model = kwargs.get('model', None)
                if model is not None:
                    best_model_path = os.path.join(self.save_dir, "best_model.pth")
                    torch.save(model.state_dict(), best_model_path)
                    print(f"\n[Monitor] 🌟 Eval Loss 進步 ({old_best:.4f} -> {current_eval_loss:.4f})，已淬鍊權重至 {best_model_path}")

# ==========================================
# 2. 精確解凍與梯度強心針機制
# ==========================================
def unfreeze_routing_components(model):
    trainable_params = 0
    total_params = 0
    
    for name, param in model.named_parameters():
        total_params += param.numel()
        param.requires_grad = False 
        
        unlock_keywords = [
            "gate_fft", "gate_mem",       
            "omegas", "mlp_H", "fft_norm",
            "Q_mem", "W_k_mem", "W_v_mem", "mem_norm", 
            "W_q_cross", "W_k_cross", "W_v_cross", "o_proj_cross" 
        ]
        
        if any(keyword in name for keyword in unlock_keywords):
            param.requires_grad = True
            trainable_params += param.numel()
            
    print(f"🔓 精確解凍完成！總參數: {total_params:,} | 路由參數: {trainable_params:,} ({(trainable_params/total_params)*100:.2f}%)")
    return model

def force_hidden_states_grad(module, inputs, output):
    """
    🎯 核心突破口：梯度強心針
    攔截 Embedding 層的輸出，強行賦予梯度追蹤能力。
    破解因 Embedding 被凍結而導致 PyTorch Checkpoint 直接截斷反向傳播的盲區。
    """
    if not output.requires_grad:
        output.requires_grad_(True)
    return output

# ==========================================
# 3. 對比學習 (喚醒內建 Chunked Loss)
# ==========================================
class TripletDataCollator:
    def __call__(self, features):
        batch = {}
        for key in ["query_ids", "pos_ids", "neg_ids"]:
            if key in features[0]:
                batch[key] = torch.tensor([f[key] for f in features], dtype=torch.long)
        return batch

class RouterTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        pos_ids = inputs.get("pos_ids")
        neg_ids = inputs.get("neg_ids")

        with torch.no_grad():
            neg_outputs = model(input_ids=neg_ids, labels=neg_ids)
            neg_loss_scalar = neg_outputs["loss"].detach()

        torch.cuda.empty_cache()

        pos_outputs = model(input_ids=pos_ids, labels=pos_ids)
        pos_loss = pos_outputs["loss"]

        margin = 1.0
        contrastive_loss = torch.clamp(pos_loss - neg_loss_scalar + margin, min=0.0)
        loss = pos_loss + 0.5 * contrastive_loss

        return (loss, pos_outputs) if return_outputs else loss

# ==========================================
# 4. 主訓練流程
# ==========================================
def main():
    model_id = "google/gemma-3-1b-it"
    data_dir = "./agiv2_router_32k_triplets"
    save_dir = "./agiv2_zerogate_checkpoints"
    
    os.makedirs(save_dir, exist_ok=True)
    
    print("📦 載入 32K ZeroGate 觸發資料集...")
    ds = load_from_disk(data_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    
    print("⚙️ 組裝 AGIV2G 主幹與 Gemma 權重...")
    base = AGIV2G(vocab_size=262144, D=1152, hidden_dim=6912, num_blocks=26)
    base = transplant_and_freeze(model_id, base)
    
    # 🌟 掛載梯度強心針到 base_model 的 embedding 層
    if hasattr(base, 'embedding'):
        base.embedding.register_forward_hook(force_hidden_states_grad)
        print("💉 已成功掛載「梯度強心針」於 Embedding 層，打通 Autograd 脈絡！")
    
    model = AGIV2GForCausalLM(base, use_gc=True)
    model = unfreeze_routing_components(model)
    
    # 注意：我們移除了舊版的 inject_gradient_checkpointing，因為它會與 use_gc=True 產生雙重切塊衝突
    
    model = model.cuda().to(torch.bfloat16)

    args = TrainingArguments(
        output_dir=save_dir,
        num_train_epochs=3,
        per_device_train_batch_size=1, 
        gradient_accumulation_steps=16, 
        learning_rate=2e-4, 
        lr_scheduler_type="cosine",
        warmup_steps=100, 
        weight_decay=0.01,
        bf16=True,
        logging_steps=1,             
        save_steps=10,               
        save_total_limit=2,          
        eval_strategy="steps",
        eval_steps=200,              
        optim="paged_adamw_8bit", 
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=4     
    )

    trainer = RouterTrainer(
        model=model,
        args=args,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        data_collator=TripletDataCollator(),
        callbacks=[ZeroGateMonitor(save_dir=save_dir)]
    )

    last_checkpoint = get_last_checkpoint(save_dir)
    if last_checkpoint is not None:
        print(f"\n🚀 偵測到極限中斷點 {last_checkpoint}，啟動全維度狀態無損恢復...")
    else:
        print("\n🚀 開始 ZeroGate 內源路由訓練！觀察蟲洞是否開啟...")

    trainer.train(resume_from_checkpoint=last_checkpoint)
    
    torch.save(model.state_dict(), os.path.join(save_dir, "zerogate_final.pth"))
    print("🎉 訓練完成！您的 AGIV2G 已成功自我演化出全局路由能力。")

if __name__ == "__main__":
    main()