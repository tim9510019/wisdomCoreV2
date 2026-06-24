"""
trainDNA16B_agentic_grpo.py — DNA-Helix 16B Coder 代理自主糾錯強化學習（GRPO）訓練引擎
================================================================================
功能：
  1. 載入 Phase 2 基礎蒸餾完成後之 DNA-Helix 16B 模型權重。
  2. 載入具備「提示詞 + 單元測試套件」的 Agentic 任務數據集。
  3. 藉由多進程 DDP (Distributed Data Parallel) 支持大規模分佈式訓練。
  4. 實作多維度沙盒獎勵評估與 GRPO (Group Relative Policy Optimization) 策略更新。
  5. 內化「思考 <think> -> 工具/代碼執行 -> 報錯自我糾錯」的自主工作流。
"""

import os
import re
import sys
import csv
import time
import math
import glob
import tempfile
import subprocess
import threading
from typing import List, Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler

from transformers import AutoTokenizer, set_seed
from datasets import load_dataset

# ============================================================================
# 全局配置
# ============================================================================
RANDOM_SEED = 2026
set_seed(RANDOM_SEED)

# ── 路徑配置 ──
DISTILL_CKPT_PATH = os.path.expanduser("~/task06_checkpoints/best_distill_model.pth")
DATASET_DIR_AGENT = os.path.expanduser("~/task06_data/agentic_tasks/") # 包含 Prompt 與 Test Suite 的 Parquet 檔案
SAVE_DIR          = os.path.expanduser("~/task06_grpo_checkpoints/")
LOG_PATH          = os.path.expanduser("~/task06_grpo_log.csv")
TOKENIZER_ID      = "Qwen/Qwen3-Coder-30B-A3B-Instruct"

# ── GRPO 超參數 ──
GROUP_SIZE        = 4      # K: 每個 Prompt 採樣的個數
EPS_CLIP          = 0.2    # PPO 剪裁區間
BETA_KL           = 0.04   # KL 散度懲罰權重
LEARNING_RATE     = 1e-6   # RL 學習率宜偏小
WEIGHT_DECAY      = 0.01
MAX_STEPS         = 2000
LOGGING_STEPS     = 1
SAVE_STEPS        = 50
EVAL_STEPS        = 50

# ============================================================================
# 1. 分佈式環境初始化 (DDP Setup)
# ============================================================================
def setup_dist() -> Tuple[int, int]:
    if not dist.is_available() or not dist.is_initialized():
        # 支援 torchrun 自動配置環境變量
        rank = int(os.environ.get("RANK", 0))
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = "localhost"
        if "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = "29500"
            
        dist.init_process_group(
            backend="nccl" if torch.cuda.is_available() else "gloo",
            init_method="env://",
            world_size=world_size,
            rank=rank
        )
        torch.cuda.set_device(local_rank)
        return rank, local_rank
    return dist.get_rank(), int(os.environ.get("LOCAL_RANK", 0))

def cleanup_dist():
    if dist.is_initialized():
        dist.destroy_process_group()

# ============================================================================
# 2. 安全代碼測試沙盒 (PythonSandbox)
# ============================================================================
class PythonSandbox:
    def __init__(self, timeout_seconds: float = 3.0):
        self.timeout = timeout_seconds

    def run_code(self, code: str, test_suite: str = "") -> Dict[str, Any]:
        """
        在獨立的子進程中執行代碼與測試套件，捕獲執行結果。
        """
        full_code = f"{code}\n\n# ─── TEST SUITE ───\n{test_suite}"
        
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w", encoding="utf-8") as temp_file:
            temp_file.write(full_code)
            temp_path = temp_file.name

        try:
            result = subprocess.run(
                [sys.executable, temp_path],
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            return {
                "exit_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "success": result.returncode == 0
            }
        except subprocess.TimeoutExpired:
            return {
                "exit_code": -1,
                "stdout": "",
                "stderr": f"TimeoutError: Exceeded {self.timeout}s limit.",
                "success": False
            }
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

# ============================================================================
# 3. 智能代理反饋與獎勵解析器 (AgenticRewardManager)
# ============================================================================
class AgenticRewardManager:
    def __init__(self):
        self.sandbox = PythonSandbox()

    def parse_and_evaluate(self, trajectory: str, test_suite: str) -> Tuple[float, Dict[str, float]]:
        """
        多維度打分，促進模型自主形成 `<think> -> 試錯 -> 糾錯 -> 通過` 的本能。
        """
        rewards = {
            "format": 0.0,      # Markdown 格式正確度
            "think": 0.0,       # 思考過程完整度
            "compile": 0.0,     # 編譯通過率
            "test_pass": 0.0,   # 單元測試通過率
            "efficiency": 0.0   # 步數與長度懲罰
        }

        # 1. 檢測思考區塊格式
        has_think_start = "<think>" in trajectory
        has_think_end = "</think>" in trajectory
        if has_think_start and has_think_end:
            rewards["think"] = 0.3
            think_content = trajectory.split("<think>")[1].split("</think>")[0].strip()
            if len(think_content) > 30:
                rewards["think"] += 0.2
            if len(think_content) > 1500: # 懲罰廢話
                rewards["efficiency"] -= 0.15
        else:
            rewards["think"] = -1.0  # 強烈懲罰不思考

        # 2. 提取 Python 代碼塊
        code_blocks = re.findall(r"```python\n(.*?)```", trajectory, re.DOTALL)
        if not code_blocks:
            code_blocks = re.findall(r"```\n(.*?)```", trajectory, re.DOTALL)

        if code_blocks:
            rewards["format"] += 0.3
            last_code = code_blocks[-1]
            
            # 3. 運行沙盒評估
            res = self.sandbox.run_code(last_code, test_suite)
            if res["success"]:
                rewards["compile"] = 0.5
                rewards["test_pass"] = 1.0  # 通過單元測試
            else:
                rewards["compile"] = 0.1
                if "AssertionError" in res["stderr"]:
                    rewards["compile"] = 0.4  # 能執行但斷言失敗，代表有邏輯自糾錯的空間
                    rewards["test_pass"] = 0.0
                else:
                    rewards["compile"] = -0.3 # 語法崩潰錯誤
        else:
            rewards["format"] = -0.5

        total_reward = sum(rewards.values())
        return total_reward, rewards

# ============================================================================
# 4. 代理任務數據集 (AgenticTaskDataset)
# ============================================================================
class AgenticTaskDataset(Dataset):
    """
    提供强化學習任務的數據集，包含 Prompt 與 Test Suite。
    如果路徑下無數據，則會自動生成一組高質量的經典代碼糾錯與工具調用任務作為種子。
    """
    def __init__(self, dataset_dir: str):
        self.tasks = []
        parquet_files = sorted(glob.glob(os.path.join(dataset_dir, "*.parquet")))
        
        if parquet_files:
            ds = load_dataset("parquet", data_files=parquet_files, split="train")
            for item in ds:
                self.tasks.append({
                    "prompt": item["prompt"],
                    "test_suite": item["test_suite"]
                })
        else:
            # 自動生成種子任務，確保訓練腳本開箱即可自洽運行
            print(f"   [WARN] 找不到 {dataset_dir}，自動生成預設 Agentic 種子任務...")
            self._build_seed_tasks()

    def _build_seed_tasks(self):
        # 種子任務 1: 除零保護
        self.tasks.append({
            "prompt": "User: Write a python function `safe_divide(a, b)` that divides a by b. If b is 0, return 0.0.\nAssistant: <think>",
            "test_suite": "assert safe_divide(10, 2) == 5.0\nassert safe_divide(5, 0) == 0.0\nassert safe_divide(-9, 3) == -3.0"
        })
        # 種子任務 2: 列表去重並排序
        self.tasks.append({
            "prompt": "User: Write a python function `unique_sorted(lst)` that returns a list containing unique elements from `lst`, sorted in ascending order.\nAssistant: <think>",
            "test_suite": "assert unique_sorted([3, 1, 2, 3, 2]) == [1, 2, 3]\nassert unique_sorted([]) == []"
        })
        # 種子任務 3: 提取並轉換數字
        self.tasks.append({
            "prompt": "User: Write a python function `extract_numbers(text)` that finds all integers in a string and returns them as a list of integers.\nAssistant: <think>",
            "test_suite": "assert extract_numbers('hello 123 world 45') == [123, 45]\nassert extract_numbers('no numbers') == []"
        })

    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, idx):
        return self.tasks[idx]

# ============================================================================
# 5. 訓練模型封裝與主體 (DNAHelixGRPOEngine)
# ============================================================================
class DNAHelixGRPOEngine(nn.Module):
    """
    包裝 DNA-Helix 模型，便於進行分佈式 forward 與梯度回傳。
    """
    def __init__(self, base_model, use_gc: bool = True):
        super().__init__()
        self.base_model = base_model
        self.use_gc = use_gc

    def forward(self, input_ids, attention_mask=None, n_split_index=None):
        import torch.utils.checkpoint as cp
        hidden_states = self.base_model.embedding(input_ids)
        hidden_states = hidden_states * math.sqrt(self.base_model.D)

        for i, block in enumerate(self.base_model.blocks):
            shift_size = (block.C // 2) if (i % 2 != 0) else 0
            if self.training and self.use_gc:
                hidden_states, _ = cp.checkpoint(
                    block, hidden_states, shift_size,
                    use_reentrant=False, n_split_index=n_split_index,
                )
            else:
                hidden_states, _ = block(
                    hidden_states, shift_size=shift_size, n_split_index=n_split_index
                )

        hidden_states = self.base_model.final_norm(hidden_states)
        logits = self.base_model.fc_out(hidden_states)
        # 與蒸餾腳本相同的雙曲正切 logits 縮放，防數值溢出
        logits = logits / 30.0
        logits = torch.tanh(logits) * 30.0
        return logits

# ============================================================================
# 6. 主程序
# ============================================================================
def main():
    rank, local_rank = setup_dist()
    is_main = (rank == 0)

    if is_main:
        print("🧬 DNA Helix Ultimate 16B Agentic-GRPO 強化學習引擎啟動...")
        print(f"   DDP World Size = {dist.get_world_size()} | Rank = {rank}")

    # ── 1. 載入 Tokenizer ──
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID, trust_remote_code=True)
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    # ── 2. 載入訓練與測試任務 ──
    os.makedirs(DATASET_DIR_AGENT, exist_ok=True)
    dataset = AgenticTaskDataset(DATASET_DIR_AGENT)
    sampler = DistributedSampler(dataset, num_replicas=dist.get_world_size(), rank=rank, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=1, sampler=sampler)

    # ── 3. 初始化 16B 模型 ──
    if is_main:
        print("\n🧬 初始化 16B DNA Helix Ultimate 模型...")
    try:
        from GEMMA3 import DNAHelixUltimate as GEMMA3
        base = GEMMA3(
            vocab_size=151936,
            D=4096,
            C=2048,
            hidden_dim=16384,
            num_blocks=40,
            rope_local=10000.0,
            rope_global=1000000.0,
        )
    except ImportError:
        raise ImportError("請確認 GEMMA3.py 存在於當前目錄下")

    # ── 4. 載入 Phase 2 基礎蒸餾權重 ──
    if os.path.exists(DISTILL_CKPT_PATH):
        if is_main:
            print(f"   [INFO] 載入基礎蒸餾權重：{DISTILL_CKPT_PATH}")
        state_dict = torch.load(DISTILL_CKPT_PATH, map_location="cpu")
        # 移除包裝前綴（若有）
        clean_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("base_model."):
                clean_state_dict[k.replace("base_model.", "")] = v
            else:
                clean_state_dict[k] = v
        base.load_state_dict(clean_state_dict, strict=False)
    else:
        if is_main:
            print(f"   [WARN] 找不到基礎蒸餾權重：{DISTILL_CKPT_PATH}，將從隨機初始化開始訓練！")

    # 封裝 active 模型與 reference 模型
    model = DNAHelixGRPOEngine(base, use_gc=True).to(torch.bfloat16).cuda()
    
    # Reference model 保持凍結，不更新參數
    import copy
    ref_base = copy.deepcopy(base)
    ref_model = DNAHelixGRPOEngine(ref_base, use_gc=False).to(torch.bfloat16).cuda()
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    # DDP 包裝 active 模型
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    reward_manager = AgenticRewardManager()

    # 初始化日誌
    if is_main:
        os.makedirs(SAVE_DIR, exist_ok=True)
        if not os.path.exists(LOG_PATH):
            with open(LOG_PATH, "w", newline="") as f:
                csv.writer(f).writerow(["step", "loss", "mean_reward", "max_reward", "kl_loss", "time"])

    # ── 5. GRPO 訓練迴圈 ──
    global_step = 0
    epoch = 0

    while global_step < MAX_STEPS:
        sampler.set_epoch(epoch)
        for batch in dataloader:
            if global_step >= MAX_STEPS:
                break

            prompt = batch["prompt"][0]
            test_suite = batch["test_suite"][0]

            # A. 生成 K 個軌跡 (僅在 DDP 各卡獨立進行以保障採樣多樣性)
            model.eval()
            inputs = tokenizer(prompt, return_tensors="pt").to(f"cuda:{local_rank}")
            prompt_len = inputs["input_ids"].size(1)

            completions_ids = []
            trajectories = []
            
            with torch.no_grad():
                for _ in range(GROUP_SIZE):
                    gen_out = model.module.base_model.generate(
                        **inputs,
                        max_new_tokens=512,
                        do_sample=True,
                        temperature=0.8,
                        top_p=0.95,
                        pad_token_id=pad_id
                    )
                    completions_ids.append(gen_out[0])
                    gen_text = tokenizer.decode(gen_out[0][prompt_len:], skip_special_tokens=True)
                    trajectories.append(gen_text)

            # B. 通過沙盒與規則評估 Reward
            rewards = []
            for traj in trajectories:
                reward, _ = reward_manager.parse_and_evaluate(traj, test_suite)
                rewards.append(reward)

            rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device="cuda")
            
            # C. 群體標準化相對優勢 (Group Standardized Advantage)
            mean_r = rewards_tensor.mean()
            std_r = rewards_tensor.std() + 1e-8
            advantages = (rewards_tensor - mean_r) / std_r

            # D. 對齊與 Padding
            max_len = max(len(c) for c in completions_ids)
            padded_inputs = []
            padded_labels = []
            attention_masks = []

            for c in completions_ids:
                pad_len = max_len - len(c)
                p_in = torch.cat([c, torch.full((pad_len,), pad_id, dtype=torch.long, device="cuda")])
                p_lbl = p_in.clone()
                p_lbl[:prompt_len] = -100
                p_lbl[len(c):] = -100
                att_mask = p_in.ne(pad_id).long()

                padded_inputs.append(p_in)
                padded_labels.append(p_lbl)
                attention_masks.append(att_mask)

            batch_inputs = torch.stack(padded_inputs)
            batch_labels = torch.stack(padded_labels)
            batch_masks = torch.stack(attention_masks)

            # E. Forward 計算 Log Probs 與 Loss
            model.train()
            
            # 1. 計算 Active 模型的 log_probs
            logits = model(batch_inputs, attention_mask=batch_masks)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = batch_labels[..., 1:].contiguous()
            log_probs = F.log_softmax(shift_logits, dim=-1)
            current_log_probs = torch.gather(log_probs, dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
            current_log_probs = (current_log_probs * (shift_labels != -100).float()).sum(dim=-1)

            # 2. 計算 Reference 模型的 log_probs (不進行梯度回傳)
            with torch.no_grad():
                ref_logits = ref_model(batch_inputs, attention_mask=batch_masks)
                ref_shift_logits = ref_logits[..., :-1, :].contiguous()
                ref_log_probs = F.log_softmax(ref_shift_logits, dim=-1)
                ref_current_log_probs = torch.gather(ref_log_probs, dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
                ref_current_log_probs = (ref_current_log_probs * (shift_labels != -100).float()).sum(dim=-1)

            # F. 計算 GRPO Policy Loss
            ratio = torch.exp(current_log_probs - ref_current_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - EPS_CLIP, 1.0 + EPS_CLIP) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # KL 懲罰 (避免 Policy 與 Ref 偏離過大)
            kl_loss = F.kl_div(
                current_log_probs, 
                ref_current_log_probs, 
                log_target=True, 
                reduction="batchmean"
            )

            loss = policy_loss + BETA_KL * kl_loss

            # G. 反向傳播與更新
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            global_step += 1

            # H. 日誌紀錄與保存
            if is_main and (global_step % LOGGING_STEPS == 0):
                print(f"[Step {global_step}] Loss={loss.item():.4f} | MeanR={mean_r.item():.4f} | MaxR={rewards_tensor.max().item():.4f} | KL={kl_loss.item():.4f}")
                with open(LOG_PATH, "a", newline="") as f:
                    csv.writer(f).writerow([
                        global_step,
                        f"{loss.item():.6f}",
                        f"{mean_r.item():.4f}",
                        f"{rewards_tensor.max().item():.4f}",
                        f"{kl_loss.item():.6f}",
                        time.ctime()
                    ])

            if is_main and (global_step % SAVE_STEPS == 0):
                ckpt_path = os.path.join(SAVE_DIR, f"dna16b_grpo_step{global_step}.pth")
                raw_model = model.module if hasattr(model, "module") else model
                torch.save(raw_model.state_dict(), ckpt_path)
                print(f"💾 Checkpoint 已保存至：{ckpt_path}")

            # DDP 同步阻斷
            dist.barrier()

        epoch += 1

    if is_main:
        final_path = os.path.join(SAVE_DIR, "dna16b_grpo_final.pth")
        raw_model = model.module if hasattr(model, "module") else model
        torch.save(raw_model.state_dict(), final_path)
        print(f"\n✅ GRPO 強化學習訓練完成！模型已保存至：{final_path}")

    cleanup_dist()

if __name__ == "__main__":
    main()
