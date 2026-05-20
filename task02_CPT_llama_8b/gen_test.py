import os
import torch
from transformers import AutoTokenizer

# 強制使用與訓練相同的環境變數
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["BITSANDBYTES_NOWELCOME"] = "1"

from train import AGIV2LForCausalLM, transplant_and_freeze
from AGIV2L import AGIV2L


def test_generation():
    model_id = "NousResearch/Meta-Llama-3-8B"

    print("⏳ [1/3] 載入 Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("⏳ [2/3] 初始化 AGIV2L 模型並掛載 Llama-3 權重...")
    base = AGIV2L(vocab_size=len(tokenizer), D=4096, hidden_dim=14336, num_blocks=32)
    base = transplant_and_freeze(model_id, base)

    model = AGIV2LForCausalLM(base).to(torch.bfloat16).cuda()
    model.eval()  # 關閉 Dropout，確保生成穩定

    # 測試提示詞 (Prompt)
    prompt = "The capital of France is Paris. The capital of Japan is"
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].cuda()

    print(f"\n⏳ [3/3] 開始生成 (Prompt: '{prompt}')...")
    print("-" * 50)
    print(prompt, end="")  # 先印出 prompt

    max_new_tokens = 10

    # 最純粹的 Greedy Decoding (貪婪解碼) 迴圈
    for _ in range(max_new_tokens):
        with torch.no_grad():
            # 取得目前的 logits (只拿最後一個 token 的預測結果)
            outputs = model(input_ids=input_ids)
            next_token_logits = outputs["logits"][:, -1, :]

            # 找出機率最大的一個字
            next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)

            # 將新預測的字接回 input_ids，準備預測下一個字
            input_ids = torch.cat([input_ids, next_token_id], dim=-1)

            # 印出這個新預測的字
            next_word = tokenizer.decode(next_token_id[0])
            print(next_word, end="", flush=True)

            # 如果預測出 EOS token，代表模型認為句子結束了
            if next_token_id.item() == tokenizer.eos_token_id:
                break

    print("\n" + "-" * 50)
    print("🎯 生成測試完畢。")


if __name__ == "__main__":
    test_generation()
