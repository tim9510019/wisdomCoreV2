"""
genCPTACV3.py — AGIV3 CPT 階段解碼 (Byte Latent 漏斗架構觀察版)
=========================================================================
核心機制：
1. 純 Byte 態解碼：詞表大小 = 256，無需 Tokenizer，逐 Byte AR 生成。
2. 宏觀鎖定：每次前向傳播強制傳入 n_split_index = len(bytes_past)，
   確保 FFT / Memory GlobalBlock 絕對不被新生成的 Byte 污染。
3. 漏斗尺度調整：GlobalBlock 接收的 adj_n_split = n_split // 4
   (因 Conv1d stride=4 的 collapse 壓縮)。
4. 動態路由觀測：從 main_blocks 中的 AGIV2GlobalBlock 擷取即時閘門配額。
5. 負載控制：MAX_PROMPT_BYTES 限制最大輸入長度，避免表徵崩潰。
"""

import os
import sys
import torch
import torch.nn.functional as F
from datasets import load_dataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from AGIV3 import AGIV3
from utils import AGIV3ForCausalLM

# ==========================================
# [ 全局配置區 ] 統一調控台
# ==========================================
# --- 1. 路徑與基礎設定 ---
CHECKPOINT_PATH = "./agiv3_cpt_checkpoints_1KSGT/best_cpt_model.pth"
DATASET_PATH = "./agiv2_stage1_1K/agiv3_raw_text.parquet"
RANDOM_SEED = 2026
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TEST_SIZE = 50  # 驗證集切割大小

# --- 2. 推論解碼參數 ---
TEST_SAMPLE_INDEX = 0  # 🎯 指定要測試的驗證集樣本 (範圍: 0 ~ TEST_SIZE-1)
MAX_DISPLAY_BYTES = 256  # 最大生成 Byte 數量
TEMPERATURE = 0.8  # 生成溫度
REPETITION_PENALTY = 1.2  # 重複懲罰係數
MAX_PROMPT_BYTES = 1024  # 限制模型最大輸入長度 (bytes)

# --- 3. AGIV3 模型物理架構參數 (需與訓練時一致) ---
USE_GC = False  # 推論時關閉 Gradient Checkpointing

# ==========================================
# [ 核心函式區 ]
# ==========================================


def load_v3_model():
    print("🧬 [載入] 正在初始化 AGIV3 Byte Latent 漏斗架構...")

    base = AGIV3()
    base.set_temperature(1.0)  # 推論用標準溫度

    model = AGIV3ForCausalLM(base, use_gc=USE_GC)

    if os.path.exists(CHECKPOINT_PATH):
        print(f"📬 [權重] 正在載入 CPT checkpoint: {CHECKPOINT_PATH}")
        state_dict = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        model.load_state_dict(state_dict, strict=True)
        print("✅ CPT 權重對齊完成。")
    else:
        print(f"⚠️ 找不到權重檔：{CHECKPOINT_PATH}，請確認訓練已產出檔案。")
        sys.exit(1)

    return model.to(DEVICE).to(torch.bfloat16).eval()


@torch.no_grad()
def generate_v3_autoregressive(
    model, prompt_bytes, max_display, temperature, repetition_penalty
):
    """
    純 Byte AR 解碼：M_global 鎖死於 N (bytes_past)，逐 Byte 自迴歸生成。
    """
    n_split = len(prompt_bytes)
    current_ids = torch.tensor([prompt_bytes], dtype=torch.long, device=DEVICE)
    n_split_tensor = torch.tensor([n_split], dtype=torch.long, device=DEVICE)

    print(f"\n🚀 [Byte AR 解碼] 啟動 (初始長度 N={n_split} bytes)")
    print(f"✨ [狀態] FFT/Memory GlobalBlock 已鎖定於前 {n_split} bytes")
    print(
        f"✨ [參數] Temperature={temperature}, Repetition_Penalty={repetition_penalty}"
    )
    print(f"✨ [狀態] Byte AR 生成引擎啟動...\n")

    generated_bytes = []
    avg_conf = 0.0

    for step in range(max_display):
        outputs = model(input_ids=current_ids, n_split_index=n_split_tensor)

        # 🔍 第一步觀測：顯示 GlobalBlock 動態路由配額
        if step == 0:
            print("🔍 [分析] 🧠 AGIV3 GlobalBlock 動態路由分配 (Byte 態納許均衡):")
            core = model.base_model
            for i, block in enumerate(core.main_blocks):
                if hasattr(block, "avg_g_loc"):
                    loc = block.avg_g_loc.item()
                    mem = block.avg_g_mem.item()
                    fft = block.avg_g_fft.item()
                    dom = (
                        "Loc"
                        if loc > mem and loc > fft
                        else ("Mem" if mem > fft else "FFT")
                    )
                    print(
                        f"  ▶ MainBlock {i:02d} | Loc: {loc:.4f} | Mem: {mem:.4f} | FFT: {fft:.4f}  [主導: {dom}]"
                    )
            print("-" * 60)

        # 取得 logits
        if isinstance(outputs, dict):
            logits = outputs["logits"]
        else:
            logits = outputs

        next_byte_logits = logits[0, -1, :].clone()  # shape: [256]

        # 重複懲罰
        if repetition_penalty > 1.0 and len(generated_bytes) > 0:
            for byte_id in set(generated_bytes):
                if next_byte_logits[byte_id] < 0:
                    next_byte_logits[byte_id] *= repetition_penalty
                else:
                    next_byte_logits[byte_id] /= repetition_penalty

        # 溫度採樣
        if temperature > 0.0:
            probs = F.softmax(next_byte_logits / temperature, dim=-1)
            next_byte = torch.multinomial(probs, num_samples=1)
            next_byte_id = next_byte.item()
            avg_conf += probs[next_byte_id].item()
        else:
            probs = F.softmax(next_byte_logits, dim=-1)
            max_prob, next_byte = torch.max(probs, dim=-1)
            next_byte_id = next_byte.item()
            avg_conf += max_prob.item()

        # EOS: byte 值 0 視為終止 (padding)
        if next_byte_id == 0:
            print(f"[Step {step+1}] 收到 0x00 (EOS)，生成終止。")
            break

        generated_bytes.append(next_byte_id)
        current_ids = torch.cat(
            [current_ids, torch.tensor([[next_byte_id]], device=DEVICE)], dim=1
        )

    avg_conf = avg_conf / max(1, len(generated_bytes))

    # 安全解碼：bytes → UTF-8（容忍非法序列）
    output_text = bytes(generated_bytes).decode("utf-8", errors="replace")

    print(f"📊 預測平均信心度: {avg_conf:.4f}")
    print(f"✅ 模型 Byte Latent 輸出：\n{output_text}")
    print("-" * 60)

    return output_text


def get_eval_sample(sample_index):
    if not os.path.exists(DATASET_PATH):
        print(f"⚠️ 找不到 V3 Parquet 資料集: {DATASET_PATH}")
        sys.exit(1)

    dataset_dict = load_dataset("parquet", data_files=DATASET_PATH)
    dataset = dataset_dict["train"].train_test_split(
        test_size=TEST_SIZE, seed=RANDOM_SEED
    )
    eval_ds = dataset["test"]

    if sample_index < 0 or sample_index >= len(eval_ds):
        print(f"⚠️ 警告: 指定的樣本 ({sample_index}) 超出範圍，重置為 0。")
        sample_index = 0

    print(
        f"📦 [資料集] 成功提取獨立驗證集 第 {sample_index} 筆樣本 (總數: {len(eval_ds)})"
    )
    return eval_ds[sample_index]


if __name__ == "__main__":
    sample = get_eval_sample(TEST_SAMPLE_INDEX)

    # V3 資料格式：text_past / text_future (raw string)
    text_past = sample["text_past"]
    text_future = sample["text_future"]

    bytes_past = list(text_past.encode("utf-8"))
    bytes_future = list(text_future.encode("utf-8"))

    # 負載截斷保護（從尾端保留，保留最近的上下文）
    if len(bytes_past) > MAX_PROMPT_BYTES:
        print(
            f"✂️ [注意] 原始 Past ({len(bytes_past)} bytes) 超載，"
            f"自動截斷保留後 {MAX_PROMPT_BYTES} bytes。"
        )
        bytes_past = bytes_past[-MAX_PROMPT_BYTES:]

    # 顯示上下文
    print(f"\n📥 [輸入上下文 text_past] (長度: {len(bytes_past)} bytes)：")
    print(
        text_past[-MAX_PROMPT_BYTES:]
        if len(text_past.encode()) > MAX_PROMPT_BYTES
        else text_past
    )
    print("-" * 60)

    # 顯示標準答案
    true_future_text = bytes(bytes_future[:MAX_DISPLAY_BYTES]).decode(
        "utf-8", errors="replace"
    )
    print(f"💡 [標準答案 text_future] (前 {MAX_DISPLAY_BYTES} bytes)：")
    print(true_future_text)
    print("-" * 60)

    print("🎯 測試啟動：AGIV3 Byte Latent CPT 自迴歸解碼驗證")

    # 載入模型
    model = load_v3_model()

    # 啟動生成
    generate_v3_autoregressive(
        model=model,
        prompt_bytes=bytes_past,
        max_display=MAX_DISPLAY_BYTES,
        temperature=TEMPERATURE,
        repetition_penalty=REPETITION_PENALTY,
    )
