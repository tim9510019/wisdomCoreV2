import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
import transformers

transformers.logging.set_verbosity_error()

# GPU 1 對齊 train.py
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def main():
    model_id = "NousResearch/Meta-Llama-3-8B"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"🔄 正在初始化 Tokenizer ({model_id})...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"🔄 正在載入原始開源模型 ({model_id}) ...")
    # 使用 Hugging Face 官方 AutoModel 載入原始權重
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map=device
    )

    # 使用 Hugging Face 的 TextStreamer 輕鬆達成打字機效果
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    print("=========================================================")
    print("🎉 Llama 原始模型測試就緒！輸入你的對話 (輸入 'quit' 或 'exit' 離開)")
    print("=========================================================")

    while True:
        try:
            user_input = input("\n👤 提問: ")
            if user_input.strip().lower() in ["quit", "exit"]:
                print("👋 掰掰！")
                break
            if not user_input.strip():
                continue

            # Llama 3 常見的指示格式，若不是 Instruct 則手動組合提示詞
            try:
                messages = [{"role": "user", "content": user_input}]
                prompt_str = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                # 若 Base 版沒有預設 Chat Template，退回到最基礎 QA 格式
                prompt_str = f"User: {user_input}\nAssistant: "

            input_ids = tokenizer(prompt_str, return_tensors="pt").input_ids.to(device)

            print("🤖 模型: ", end="", flush=True)
            _ = model.generate(
                input_ids,
                max_new_tokens=300,
                temperature=0.7,
                top_k=50,
                do_sample=True,
                streamer=streamer,
                pad_token_id=tokenizer.eos_token_id,
            )

        except KeyboardInterrupt:
            print("\n👋 掰掰！")
            break


if __name__ == "__main__":
    main()
