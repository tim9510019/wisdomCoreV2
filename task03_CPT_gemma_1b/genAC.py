"""
genAC.py — AGIV2 並行映射解碼腳本 (N -> N+B 物理隔離收斂版)
===========================================================
核心機制修正：
1. 取消自迴歸 (Non-Autoregressive)：由於 AGIV2 是雙向全域架構 (無因果遮罩)。
   在 trainRouterAC_1K 中，模型被迫在看到 [PAD, PAD...] 的時候，「同時」預測出答案的所有 Token。
   若強行使用 Token-by-Token 替換 PAD，會產生模型從未見過的 Out-of-Distribution (OOD) 狀態導致亂碼。
2. 平行拔取 (Parallel Decoding)：我們直接將輸入維持 N+B (B全為 PAD)，進行單次 Forward，
   並一次性提取對應 B 區塊範圍的所有預測結果 (Argmax)，這才是完美反映 GROK 的解碼方式！
"""

import os
import sys
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from datasets import Dataset
from utils import QuantumRouterEngineAC, DynamicACDataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from AGIV2GAC import AGIV2G

MODEL_ID = "google/gemma-3-1b-it"
CHECKPOINT_PATH = "./agiv2_zerogate_ac_checkpoints_1K/best_model.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
B_SIZE = 512


def load_ac_model():
    print(f"🧬 [載入] 正在初始化 AGIV2GAC 物理隔離架構...")
    model = AGIV2G(
        vocab_size=262144, D=1152, C=256, hidden_dim=6912, num_blocks=26, head_dim=256
    )

    if os.path.exists(CHECKPOINT_PATH):
        print(f"📬 [權重] 正在從 {CHECKPOINT_PATH} 載入最佳模型...")
        state_dict = torch.load(CHECKPOINT_PATH, map_location=DEVICE)

        new_state_dict = {}
        for k, v in state_dict.items():
            name = (
                k.replace("base_model.model.", "")
                .replace("base_model.", "")
                .replace("model.", "")
            )
            new_state_dict[name] = v

        model.load_state_dict(new_state_dict, strict=False)
        print("✅ 權重對齊完成。")
    else:
        print(f"⚠️ 找不到權重檔：{CHECKPOINT_PATH}")
        sys.exit(1)

    return model.to(DEVICE).to(torch.bfloat16).eval()


def check_routing_gates(model):
    print("\n🔍 [分析] 檢查路由閘門 (Routing Gates) 開啟狀況：")
    for i, block in enumerate(model.blocks):
        if hasattr(block, "gate_fft"):
            fft_val = block.gate_fft.item()
            mem_val = block.gate_mem.item()
            print(
                f"  ▶ Block {i:02d} | gate_fft: {fft_val:+.4f} | gate_mem: {mem_val:+.4f}"
            )
    print("-" * 50)


@torch.no_grad()
def generate_ac_robust(model, tokenizer, prompt_ids, max_gen=64):
    pad_id = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id
    )
    N_len = len(prompt_ids)

    # [核心修復 2] 直接利用 AGIV2 的非因果平行映射
    # 因為訓練時的輸入 B 區塊被隔離 (全部設為 PAD)，模型靠 M_global 直接平行輸出整個 B 區塊的 Logits
    # 只要我們取代任何一個 PAD 成為真實 Token，就會打破它訓練時只看過 [PAD] 的流形 (OOD 狀態)
    input_ids = prompt_ids + [pad_id] * B_SIZE
    current_ids = torch.tensor(input_ids, dtype=torch.long, device=DEVICE).unsqueeze(0)

    print(f"\n🚀 [並行映射] 長度 L={N_len + B_SIZE} (N={N_len} + B={B_SIZE})")
    print(
        "✨ [解碼修正] 關閉自迴歸，啟動全矩陣『單次 Forward 平行拔出』 (Single-Shot Extraction)"
    )
    print("-" * 50)

    prompt_text = tokenizer.decode(prompt_ids, skip_special_tokens=True)
    display_prompt = prompt_text[-300:]
    print(f"...{display_prompt}\n", flush=True)

    # 執行單次 Forward (與 Evaluation 時相同)
    logits = model(current_ids)

    # 根據平移量對齊，第 N_len 位置的預測來自 第 N_len - 1 的 Logit
    # 我們一口氣把整個 B_SIZE 的預測 Logits 抽取出來
    pred_logits = logits[0, N_len - 1 : N_len - 1 + B_SIZE, :]
    pred_ids_full = torch.argmax(pred_logits, dim=-1).tolist()

    # 截斷遇到 PAD 或 EOS 後的無限雜訊
    final_ids = []
    for pid in pred_ids_full:
        if pid == tokenizer.pad_token_id or pid == tokenizer.eos_token_id:
            break
        final_ids.append(pid)

    output_text = tokenizer.decode(final_ids, skip_special_tokens=True)
    print(f"✅ 平行輸出：{output_text}")
    print("-" * 50)

    return output_text


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    print(f"\n📦 [資料集] 正在初始化及時數據生成引擎...")
    engine = QuantumRouterEngineAC()
    # 我們只需要一筆測試資料，所以生成數量設為1
    test_ds = DynamicACDataset(engine, target_lengths=[1024], num_samples=1)
    print(f"✅ 成功初始化及時測試集")

    model = load_ac_model()
    check_routing_gates(model)

    # 抽取第一筆測試資料
    sample = test_ds[0]
    pos_ids = sample["pos_ids"]
    N_len = len(pos_ids) - B_SIZE
    prompt_ids = pos_ids[:N_len]

    # 取出真實答案以供對照
    true_ans_ids = pos_ids[N_len : N_len + 64]  # 取前64個非PAD供比對
    true_ans_text = tokenizer.decode(
        [i for i in true_ans_ids if i != tokenizer.pad_token_id],
        skip_special_tokens=True,
    )

    print(f"\n💡 [標準答案] B區塊內容：{true_ans_text}")
    print("\n🎯 測試項目：提取 AC_1K 真實測試資料")
    generate_ac_robust(model, tokenizer, prompt_ids, max_gen=64)
