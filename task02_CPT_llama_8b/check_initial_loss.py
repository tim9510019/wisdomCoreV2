import os
import sys

# 強制使用與訓練相同的環境變數
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["BITSANDBYTES_NOWELCOME"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from transformers import AutoTokenizer

from trainAGI import AGIV2LForCausalLM, transplant_and_freeze
from AGIV2L import AGIV2L


def verify_initial_state_with_qa():
    model_id = "NousResearch/Meta-Llama-3-8B"

    print("⏳ [1/4] 載入 Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("⏳ [2/4] 初始化 AGIV2L 模型並掛載 Llama-3 權重...")
    base = AGIV2L(vocab_size=len(tokenizer), D=4096, hidden_dim=14336, num_blocks=32)
    base = transplant_and_freeze(model_id, base)

    model = AGIV2LForCausalLM(base).to(torch.bfloat16).cuda()
    model.eval()  # 測試初始 Loss 時，關閉 Dropout 與 GC 干擾

    print("⏳ [3/4] 載入簡單邏輯與數學 QA 測試語料...")
    # 使用極度規律的內容，這對 Base 模型來說應該要有極低的 Loss (< 2.0)
    simple_qa_text = (
        "Question: What is 1 + 1?\nAnswer: 2.\n"
        "Question: What is 5 + 5?\nAnswer: 10.\n"
        "Question: What is 10 + 10?\nAnswer: 20.\n"
        "Question: What is the capital of Japan?\nAnswer: Tokyo.\n"
        "Question: What is the capital of France?\nAnswer: Paris.\n"
    )

    tokens = tokenizer(simple_qa_text, return_tensors="pt")["input_ids"].cuda()
    labels = tokens.clone()

    print(f"⏳ [4/4] 執行單步前向傳播 (序列長度: {tokens.size(1)})...")
    with torch.no_grad():
        outputs = model(input_ids=tokens, labels=labels)

    loss = outputs["loss"].item()

    # 觀測閘控初始值
    gate_fft_vals = [b.gate_fft.item() for b in base.blocks if hasattr(b, "gate_fft")]
    gate_mem_vals = [b.gate_mem.item() for b in base.blocks if hasattr(b, "gate_mem")]

    avg_gate_fft = sum(gate_fft_vals) / len(gate_fft_vals) if gate_fft_vals else 0.0
    avg_gate_mem = sum(gate_mem_vals) / len(gate_mem_vals) if gate_mem_vals else 0.0

    print("\n" + "=" * 50)
    print("🎯 物理狀態與初始條件驗證報告 (QA 測試)")
    print("=" * 50)
    print(f"🔹 測試文本: \n{simple_qa_text.strip()}")
    print("-" * 50)
    print(f"🔹 初始 Loss 值: {loss:.4f}")
    print(f"🔹 閘控 FFT 平均: {avg_gate_fft:.4f} (應為 0.0000)")
    print(f"🔹 閘控 Mem 平均: {avg_gate_mem:.4f} (應為 0.0000)")
    print("=" * 50)


if __name__ == "__main__":
    verify_initial_state_with_qa()
