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
    print("⚠️ 警告: 未安裝 python-pptx。圖表將匯出為 PNG，但無法自動生成 PPTX。請執行 `pip install python-pptx` 以啟用此功能。")

# 設定環境
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def generate_haystack_with_needle(tokenizer, target_length, depth_ratio, needle, question):
    """生成包含 Needle 的 Haystack"""
    # 填充背景文字 (Paul Graham 的文章片段作為無關背景)
    filler = "The quick brown fox jumps over the lazy dog. Programming is the art of expressing solutions to problems. "
    filler_tokens = tokenizer(filler, add_special_tokens=False).input_ids
    
    # 預算長度
    needle_tokens = tokenizer(" " + needle, add_special_tokens=False).input_ids
    question_tokens = tokenizer(f"\n{question}\nAnswer: ", add_special_tokens=False).input_ids
    
    # 需要填補的長度
    required_filler_len = max(0, target_length - len(needle_tokens) - len(question_tokens) - 5)
    
    # 構建填充序列
    repeats = required_filler_len // len(filler_tokens) + 1
    haystack_tokens = (filler_tokens * repeats)[:required_filler_len]
    
    # 插入位置 (基於 depth_ratio)
    insert_idx = int(len(haystack_tokens) * depth_ratio)
    
    # 組合: [Haystack 前半] + [Needle] + [Haystack 後半] + [Question]
    final_input_ids = haystack_tokens[:insert_idx] + needle_tokens + haystack_tokens[insert_idx:] + question_tokens
    
    # 確保不會超過 target_length 大太多
    if len(final_input_ids) > target_length + 20:
        final_input_ids = final_input_ids[:target_length]
        final_input_ids[-len(question_tokens):] = question_tokens # 確保保留問題
        
    prompt_text = tokenizer.decode(final_input_ids)
    
    # 套用 Gemma Chat Template
    messages = [{"role": "user", "content": prompt_text}]
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return formatted_prompt

def generate_custom(model, input_ids, max_new_tokens=20, eos_token_id=None):
    """給 AGIV2 的無 KV Cache 暴力遞迴生成法 (嚴格 Greed Search)"""
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
        # CPT & SFT 走 AGIV2 架構
        from train import AGIV2GForCausalLM, transplant_and_freeze
        from AGIV2G import AGIV2G
        print(f"🔄 建構 AGIV2GForCausalLM 架構...")
        base_agiv2 = AGIV2G()
        base_agiv2 = transplant_and_freeze(model_id, base_agiv2)
        model = AGIV2GForCausalLM(base_agiv2).to(torch.bfloat16).cuda()
        
        if mode == "cpt":
            cpt_path = "./agiv2-cpt/best_model.pth"
            print(f"📥 載入 CPT 權重 ({cpt_path})...")
            model.load_state_dict(torch.load(cpt_path))
        elif mode == "sft":
            sft_best_path = "./agiv2-sft/best_sft_model.pth"
            print(f"📥 載入 SFT 權重...")
            if os.path.exists(sft_best_path):
                print(f"✅ 找到最佳淬鍊模型: {sft_best_path}")
                model.load_state_dict(torch.load(sft_best_path))
            elif os.path.exists(os.path.join(checkpoint_path, "model.safetensors")):
                from safetensors.torch import load_file
                model.load_state_dict(load_file(os.path.join(checkpoint_path, "model.safetensors")))
            elif os.path.exists(os.path.join(checkpoint_path, "pytorch_model.bin")):
                model.load_state_dict(torch.load(os.path.join(checkpoint_path, "pytorch_model.bin")))
            else:
                raise FileNotFoundError(f"找不到 SFT 權重檔。請確認目錄有 {sft_best_path} 或提供了正確的 checkpoint: {checkpoint_path}")

    model.eval()
    
    lengths = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16382, 32764]
    depths = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    needle = "The secret activation code for the project is GALAXY-42."
    expected_answer = "GALAXY-42"
    question = "What is the secret activation code for the project? Answer only with the code."
    
    results = np.zeros((len(lengths), len(depths)))
    
    print("🚀 開始執行 NIAH 撈針壓力測試...")
    for i, length in enumerate(lengths):
        for j, depth in enumerate(depths):
            print(f"[{i+1}/{len(lengths)}] 測試長度: {length}, 插入深度: {depth*100:.0f}% ... ", end="")
            
            prompt = generate_haystack_with_needle(tokenizer, length, depth, needle, question)
            input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
            
            with torch.no_grad():
                if mode == "base":
                    out_ids = model.generate(
                        input_ids,
                        max_new_tokens=20,
                        pad_token_id=tokenizer.eos_token_id,
                        do_sample=False # 嚴格 Greed Search 保證測試重現度
                    )
                else:
                    # AGIV2 自己硬核生成的 fallback (沒有 generate 函數)
                    out_ids = generate_custom(model, input_ids, max_new_tokens=20, eos_token_id=tokenizer.eos_token_id)
                
            generated_ans = tokenizer.decode(out_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
            
            # 判斷命中 (有包含 GALAXY-42 即算命中，因為我們要求它只輸出代碼)
            is_correct = 1.0 if expected_answer.lower() in generated_ans.lower() else 0.0
            results[i, j] = is_correct
            
            status = "✅ 成功" if is_correct else f"❌ 失敗 (回答: {generated_ans.strip()})"
            print(status)
            
            # 清理顯存避免 OOM
            del input_ids, out_ids
            gc.collect()
            torch.cuda.empty_cache()
            
    return lengths, depths, results

def generate_reports(lengths, depths, results, mode="base"):
    os.makedirs("./niah_reports", exist_ok=True)
    
    # 1. 繪製 Heatmap (熱力圖)
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        results, 
        xticklabels=[f"{int(d*100)}%" for d in depths], 
        yticklabels=lengths,
        cmap="RdYlGn", 
        annot=True, 
        cbar=False,
        linewidths=0.5
    )
    title_prefix = "Needle In A Haystack - Accuracy Heatmap\nGoogle/Gemma-3-1b-it"
    plt.title(f"{title_prefix} [{mode.upper()}]", fontsize=16)
    plt.xlabel("Insertion Depth (%)", fontsize=12)
    plt.ylabel("Context Length (Tokens)", fontsize=12)
    heatmap_path = f"./niah_reports/niah_heatmap_{mode}.png"
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 繪製 長度與準確率折線圖
    acc_by_length = results.mean(axis=1) * 100
    plt.figure(figsize=(10, 6))
    plt.plot(lengths, acc_by_length, marker='o', linestyle='-', linewidth=2, color='blue')
    plt.xscale('log', base=10)
    plt.xticks(lengths, lengths, rotation=45)
    plt.title(f"Accuracy vs Context Length [{mode.upper()}]", fontsize=16)
    plt.xlabel("Context Length", fontsize=12)
    plt.ylabel("Average Accuracy (%)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    line_path = f"./niah_reports/niah_length_acc_{mode}.png"
    plt.savefig(line_path, dpi=300, bbox_inches='tight')
    plt.close()

    # 3. 繪製 插入深度與準確率長條圖 (Histogram)
    acc_by_depth = results.mean(axis=0) * 100
    plt.figure(figsize=(10, 6))
    plt.bar([f"{int(d*100)}%" for d in depths], acc_by_depth, color='coral', edgecolor='black')
    plt.title(f"Accuracy vs Insertion Depth [{mode.upper()}]", fontsize=16)
    plt.xlabel("Insertion Depth", fontsize=12)
    plt.ylabel("Average Accuracy (%)", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    hist_path = f"./niah_reports/niah_depth_hist_{mode}.png"
    plt.savefig(hist_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print("📊 圖表已成功儲存至 ./niah_reports 資料夾下！")
    
    # 4. 生成 PPTX 簡報
    if HAS_PPTX:
        prs = Presentation()
        
        # 標題頁
        title_slide = prs.slides.add_slide(prs.slide_layouts[0])
        title_slide.shapes.title.text = f"Gemma-3-1b-it NIAH 撈針壓力測試報告 ({mode.upper()})"
        title_slide.placeholders[1].text = f"長度範圍 32 ~ 32,764 Tokens\n模型狀態: {mode.upper()}"
        
        # 熱力圖頁
        slide = prs.slides.add_slide(prs.slide_layouts[5])
        slide.shapes.title.text = "撈針測試熱力圖 (Context Length vs Depth)"
        slide.shapes.add_picture(heatmap_path, Inches(1), Inches(1.5), height=Inches(5.5))
        
        # 長度趨勢頁
        slide2 = prs.slides.add_slide(prs.slide_layouts[5])
        slide2.shapes.title.text = "上下文長度與準確率衰退趨勢"
        slide2.shapes.add_picture(line_path, Inches(1.5), Inches(1.5), width=Inches(7))
        
        # 插入深度分佈頁
        slide3 = prs.slides.add_slide(prs.slide_layouts[5])
        slide3.shapes.title.text = "插入深度對模型遺忘率的影響 (Histogram)"
        slide3.shapes.add_picture(hist_path, Inches(1.5), Inches(1.5), width=Inches(7))
        
        pptx_path = f"./niah_reports/NIAH_Report_Gemma1B_{mode}.pptx"
        prs.save(pptx_path)
        print(f"📑 精美簡報已經生成: {pptx_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="執行 NIAH 撈針壓力測試")
    parser.add_argument("--mode", type=str, choices=["base", "cpt", "sft"], default="base",
                        help="選擇要評估的模型階段: base(原始), cpt(預訓練完成), sft(微調完成)")
    parser.add_argument("--checkpoint", type=str, default="./agiv2-sft/checkpoint-1000",
                        help="如果 mode=sft，請提供 SFT checkpoint 的資料夾路徑")
    
    args = parser.parse_args()
    
    lengths, depths, results = run_niah_test(mode=args.mode, checkpoint_path=args.checkpoint)
    generate_reports(lengths, depths, results, mode=args.mode)
