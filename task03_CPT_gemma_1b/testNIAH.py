import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForCausalLM
import gc

try:
    from pptx import Presentation
    from pptx.util import Inches
    HAS_PPTX = True
except ImportError:
    HAS_PPTX = False
    print("⚠️ 警告: 未安裝 python-pptx。圖表將匯出為 PNG，但無法自動生成 PPTX。")

# 設定環境
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def generate_haystack_with_needle(tokenizer, target_length, depth_ratio, needle, question):
    """生成包含 Needle 的 Haystack"""
    filler = "The quick brown fox jumps over the lazy dog. Programming is the art of expressing solutions to problems. "
    filler_tokens = tokenizer(filler, add_special_tokens=False).input_ids
    
    needle_tokens = tokenizer(" " + needle, add_special_tokens=False).input_ids
    question_tokens = tokenizer(f"\n{question}\nAnswer: ", add_special_tokens=False).input_ids
    
    required_filler_len = max(0, target_length - len(needle_tokens) - len(question_tokens) - 5)
    
    repeats = required_filler_len // len(filler_tokens) + 1
    haystack_tokens = (filler_tokens * repeats)[:required_filler_len]
    
    insert_idx = int(len(haystack_tokens) * depth_ratio)
    
    final_input_ids = haystack_tokens[:insert_idx] + needle_tokens + haystack_tokens[insert_idx:] + question_tokens
    
    if len(final_input_ids) > target_length + 20:
        final_input_ids = final_input_ids[:target_length]
        final_input_ids[-len(question_tokens):] = question_tokens 
        
    prompt_text = tokenizer.decode(final_input_ids)
    
    messages = [{"role": "user", "content": prompt_text}]
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return formatted_prompt

def generate_custom(model, input_ids, max_new_tokens=20, eos_token_id=None):
    """無 KV Cache 暴力生成法，確保 AGIV2G 路由正常運作"""
    for _ in range(max_new_tokens):
        outputs = model(input_ids)
        logits = outputs.get("logits", outputs) if isinstance(outputs, dict) else outputs
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        input_ids = torch.cat([input_ids, next_token], dim=-1)
        if eos_token_id is not None and next_token.item() == eos_token_id:
            break
    return input_ids

def run_niah_test(mode="base", checkpoint_path=""):
    model_id = "google/gemma-3-1b-it"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"🔄 載入 Tokenizer ({model_id})...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    
    if mode == "base":
        print(f"🔄 載入純淨 Base Model ({model_id})...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map=device
        )
    else:
        # CPT, SFT, ROUTER 均走 AGIV2 架構
        from trainAGI import AGIV2GForCausalLM, transplant_and_freeze
        from AGIV2G import AGIV2G
        print(f"🔄 建構 AGIV2GForCausalLM 架構...")
        base_agiv2 = AGIV2G()
        base_agiv2 = transplant_and_freeze(model_id, base_agiv2)
        model = AGIV2GForCausalLM(base_agiv2, use_gc=False).to(torch.bfloat16).cuda()
        
        if mode == "cpt":
            path = "./agiv2-cpt-32k/best_model.pth"
        elif mode == "router":
            path = "./agiv2_zerogate_checkpoints/best_model.pth"
            print(f"🎯 鎖定 ZeroGate 路由最佳模型...")
        elif mode == "sft":
            path = "./agiv2-sft/best_sft_model.pth"
        
        if os.path.exists(path):
            print(f"📥 載入權重 ({path})...")
            model.load_state_dict(torch.load(path))
        else:
            print(f"⚠️ 找不到路徑 {path}，嘗試從 checkpoint 載入...")
            # Fallback 邏輯... (略)

    model.eval()
    
    # 測試維度：32 ~ 32K 長度，11 種深度
    lengths = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16382, 32764]
    depths = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    needle = "The secret activation code for the project is GALAXY-42."
    expected_answer = "GALAXY-42"
    question = "What is the secret activation code for the project? Answer only with the code."
    
    results = np.zeros((len(lengths), len(depths)))
    
    print(f"🚀 開始執行 [{mode.upper()}] NIAH 撈針壓力測試...")
    for i, length in enumerate(lengths):
        for j, depth in enumerate(depths):
            print(f"[{i+1}/{len(lengths)}] 長度: {length}, 深度: {depth*100:.0f}% ... ", end="")
            
            prompt = generate_haystack_with_needle(tokenizer, length, depth, needle, question)
            input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
            
            with torch.no_grad():
                if mode == "base":
                    out_ids = model.generate(input_ids, max_new_tokens=20, do_sample=False, pad_token_id=tokenizer.eos_token_id)
                else:
                    out_ids = generate_custom(model, input_ids, max_new_tokens=20, eos_token_id=tokenizer.eos_token_id)
                
            generated_ans = tokenizer.decode(out_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
            is_correct = 1.0 if expected_answer.lower() in generated_ans.lower() else 0.0
            results[i, j] = is_correct
            
            status = "✅ 成功" if is_correct else f"❌ 失敗 (回答: {generated_ans.strip()})"
            print(status)
            
            del input_ids, out_ids
            gc.collect()
            torch.cuda.empty_cache()
            
    return lengths, depths, results

def generate_reports(lengths, depths, results, mode="base"):
    os.makedirs("./niah_reports", exist_ok=True)
    
    # 1. Heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(results, xticklabels=[f"{int(d*100)}%" for d in depths], yticklabels=lengths,
                cmap="RdYlGn", annot=True, cbar=False, linewidths=0.5)
    plt.title(f"NIAH Heatmap - Google/Gemma-3-1b-it [{mode.upper()}]", fontsize=16)
    plt.xlabel("Insertion Depth (%)")
    plt.ylabel("Context Length (Tokens)")
    heatmap_path = f"./niah_reports/niah_heatmap_{mode}.png"
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 報告已儲存至 ./niah_reports/niah_heatmap_{mode}.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="執行 NIAH 撈針壓力測試")
    parser.add_argument("--mode", type=str, choices=["base", "cpt", "sft", "router"], default="base",
                        help="選擇評估階段: base, cpt, sft, router(對比路由)")
    parser.add_argument("--checkpoint", type=str, default="./agiv2-sft/checkpoint-1000")
    
    args = parser.parse_args()
    
    lengths, depths, results = run_niah_test(mode=args.mode, checkpoint_path=args.checkpoint)
    generate_reports(lengths, depths, results, mode=args.mode)