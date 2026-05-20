"""
genCPTAC.py — AGIV2 CPT 階段解碼 (動態 MoE 路由：觀察納許均衡版)
=========================================================================
核心機制：
1. 認知對齊：訓練期已採用非對稱解耦，推論時必須採用 AR (自迴歸) 迴圈來釋放 Local SDPA 的文法能力。
2. 宏觀鎖定：每一次前向傳播皆強制傳入 n_split_index = N，確保 M_global (NAR引擎) 絕對不會被新生成的字元污染。
3. 動態路由：直接觀察模型針對當前 Input Context 所演化出的 Token-level 閘門配額 (Loc / Mem / FFT)。
4. 駐波破除：引入 Temperature 與 Repetition Penalty 打破 Greedy Decoding 造成的語義引力井。
5. 負載控制：引入 MAX_PROMPT_LENGTH 限制 1B 模型的最大注意力負載，避免表徵崩潰。
"""

import os
import sys
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from datasets import load_dataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from AGIV2GACG import AGIV2G
from utils import AGIV2GForCausalLMT

# ==========================================
# [ 全局配置區 ] 統一調控台
# ==========================================
# --- 1. 路徑與基礎設定 ---
MODEL_ID = "google/gemma-3-1b-it"
CHECKPOINT_PATH = "./agiv2_cpt_checkpoints_1KSG/best_cpt_model.pth"
DATASET_PATH = "./agiv2_stage1_1K/agiv2_stage1_N_B.parquet"
RANDOM_SEED = 2026
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TEST_SIZE = 50  # 驗證集切割大小

# --- 2. 推論解碼參數 ---
TEST_SAMPLE_INDEX = 0  # 🎯 指定要測試的驗證集樣本 (範圍: 0 ~ TEST_SIZE-1)
MAX_DISPLAY_TOKENS = 128  # 最大生成 Token 數量
TEMPERATURE = 0.1  # 生成溫度
REPETITION_PENALTY = 1.2  # 重複懲罰係數
MAX_PROMPT_LENGTH = 512  # 限制 1B 模型的最大輸入長度

# --- 3. AGIV2G 模型物理架構參數 ---
VOCAB_SIZE = 262144
D_DIM = 1152
C_DIM = 256
HIDDEN_DIM = 6912
NUM_BLOCKS = 26
HEAD_DIM = 256
USE_GC = False  # 推論時關閉 Gradient Checkpointing

# ==========================================
# [ 核心函式區 ]
# ==========================================


def load_cpt_model():
    print(f"🧬 [載入] 正在初始化 AGIV2G 物理隔離架構 (動態 MoE 路由版)...")

    base = AGIV2G(
        vocab_size=VOCAB_SIZE,
        D=D_DIM,
        C=C_DIM,
        hidden_dim=HIDDEN_DIM,
        num_blocks=NUM_BLOCKS,
        head_dim=HEAD_DIM,
    )

    # 強制將推論溫度設為 1.0 (標準納許均衡態) 或更低，確保網路表現穩定
    base.set_temperature(1.0)

    model = AGIV2GForCausalLMT(base, use_gc=USE_GC)

    if os.path.exists(CHECKPOINT_PATH):
        print(f"📬 [權重] 正在載入 CPT 結構穩定權重: {CHECKPOINT_PATH}")
        state_dict = torch.load(CHECKPOINT_PATH, map_location=DEVICE)

        model.load_state_dict(state_dict, strict=True)
        print("✅ CPT 權重對齊完成 (結構穩固：動態 Router 與 LayerNorm 已就緒)。")
    else:
        print(f"⚠️ 找不到權重檔：{CHECKPOINT_PATH}，請確認訓練已產出檔案。")
        sys.exit(1)

    return model.to(DEVICE).to(torch.bfloat16).eval()


@torch.no_grad()
def generate_cpt_autoregressive(
    model,
    tokenizer,
    prompt_ids,
    expected_b_size,
    max_display,
    temperature,
    repetition_penalty,
):
    """
    雙擎異步解碼：M_global 鎖死於 N，Local SDPA 逐字 AR 生成。
    並動態擷取模型面對 Context 時的路由決策。
    """
    N_len = len(prompt_ids)
    current_ids = torch.tensor([prompt_ids], dtype=torch.long, device=DEVICE)
    n_split_tensor = torch.tensor([N_len], dtype=torch.long, device=DEVICE)

    print(f"\n🚀 [雙擎解碼] 啟動 AR + NAR 異步融合生成 (初始長度 N={N_len})")
    print(f"✨ [狀態] 宏觀引擎 M_global (NAR) 已鎖定於前 {N_len} 個 Token。")
    print(
        f"✨ [參數] Gen_Temperature={temperature}, Repetition_Penalty={repetition_penalty}"
    )
    print(f"✨ [狀態] 微觀引擎 Local SDPA (AR) 啟動字元接龍...\n")

    generated_ids = []
    avg_conf = 0.0

    for step in range(max_display):
        # 前向傳播
        outputs = model(input_ids=current_ids, n_split_index=n_split_tensor)

        # 🌟 攔截觀測點：在第一次處理完完整 Context 時，印出動態路由比例
        if step == 0:
            print("🔍 [分析] 🧠 模型對當前 Context 的動態路由分配 (平均納許均衡):")
            core = model.base_model
            for i, block in enumerate(core.blocks):
                if hasattr(block, "avg_g_loc"):
                    loc = block.avg_g_loc.item()
                    mem = block.avg_g_mem.item()
                    fft = block.avg_g_fft.item()
                    # 凸顯最強勢的引擎
                    dom = (
                        "Loc"
                        if loc > mem and loc > fft
                        else ("Mem" if mem > fft else "FFT")
                    )
                    print(
                        f"  ▶ Block {i:02d} | Loc: {loc:.4f} | Mem: {mem:.4f} | FFT: {fft:.4f}  [主導: {dom}]"
                    )
            print("-" * 50)

        # 處理 Logits
        if hasattr(outputs, "logits"):
            logits = outputs.logits
        elif isinstance(outputs, dict):
            logits = outputs["logits"]
        else:
            logits = outputs

        next_token_logits = logits[0, -1, :].clone()

        # 重複懲罰 (Repetition Penalty)
        if repetition_penalty > 1.0 and len(generated_ids) > 0:
            for token_id in set(generated_ids):
                if next_token_logits[token_id] < 0:
                    next_token_logits[token_id] *= repetition_penalty
                else:
                    next_token_logits[token_id] /= repetition_penalty

        # 溫度與取樣
        if temperature > 0.0:
            probs = F.softmax(next_token_logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            next_token_id = next_token.item()
            avg_conf += probs[next_token_id].item()
        else:
            probs = F.softmax(next_token_logits, dim=-1)
            max_prob, next_token = torch.max(probs, dim=-1)
            next_token_id = next_token.item()
            avg_conf += max_prob.item()

        if next_token_id == tokenizer.eos_token_id:
            break

        generated_ids.append(next_token_id)
        current_ids = torch.cat(
            [current_ids, torch.tensor([[next_token_id]], device=DEVICE)], dim=1
        )

    avg_conf = avg_conf / max(1, len(generated_ids))
    output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    print(f"📊 預測平均信心度: {avg_conf:.4f}")
    print(f"✅ 模型動態路由輸出：\n{output_text}")
    print("-" * 50)

    return output_text


def get_eval_sample(sample_index):
    if not os.path.exists(DATASET_PATH):
        print(f"⚠️ 找不到實體 Parquet 資料集: {DATASET_PATH}")
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
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    sample = get_eval_sample(TEST_SAMPLE_INDEX)
    full_seq = sample["input_ids"]
    n_split = sample["n_split_index"]
    prompt_ids = full_seq[:n_split]

    # 負載截斷保護
    if len(prompt_ids) > MAX_PROMPT_LENGTH:
        print(
            f"✂️ [注意] 原始上下文 ({len(prompt_ids)}) 超載，自動截斷保留後 {MAX_PROMPT_LENGTH} 個 Token。"
        )
        prompt_ids = prompt_ids[-MAX_PROMPT_LENGTH:]

    true_b_ids = full_seq[n_split:]
    b_size = len(true_b_ids)

    # 載入模型
    model = load_cpt_model()

    # 顯示上下文
    input_text = tokenizer.decode(
        [i for i in prompt_ids if i != 0], skip_special_tokens=True
    )
    print(
        f"\n📥 [輸入上下文 Input Context] A 區塊內容 (長度: {len(prompt_ids)} Token)：\n{input_text}"
    )
    print("-" * 50)

    # 顯示標準答案
    true_ans_text = tokenizer.decode(
        [i for i in true_b_ids[:MAX_DISPLAY_TOKENS] if i != 0], skip_special_tokens=True
    )
    print(
        f"💡 [標準答案 Ground Truth] B 區塊內容 (前 {MAX_DISPLAY_TOKENS} Token)：\n{true_ans_text}"
    )
    print("-" * 50)

    print("🎯 測試啟動：AGIV2 CPT 雙擎異步解碼驗證 (動態觀測)")

    # 啟動生成
    generate_cpt_autoregressive(
        model=model,
        tokenizer=tokenizer,
        prompt_ids=prompt_ids,
        expected_b_size=b_size,
        max_display=MAX_DISPLAY_TOKENS,
        temperature=TEMPERATURE,
        repetition_penalty=REPETITION_PENALTY,
    )
