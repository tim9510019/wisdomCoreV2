"""
skill_eval_scratch_5way.py
五路 From-Scratch 模型技能評估腳本
評估五個維度：注意力熵、表示空間各向同性、長度外推、干擾魯棒性、週期結構捕捉
"""
import os, sys, csv, math, json, types
import torch
import torch.nn.functional as F
import numpy as np

sys.path.append("/mnt/home/caijingnina-gmai-11e66a/wisdomCoreV2")
sys.path.append("/mnt/home/caijingnina-gmai-11e66a/wisdomCoreV2/task03_CPT_gemma_1b")

from GEMMA3 import GEMMA3
from utils import AGIV2GForCausalLMT

CHUNK      = 256
ROPE_LOCAL = 10000.0
ROPE_GLOBAL= 1000000.0
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

# ──────────────────────────────────────────
# 架構工廠：依名稱建構對應模型
# ──────────────────────────────────────────
def build_model(arch: str, ckpt_path: str):
    base = GEMMA3(vocab_size=262144, D=1152, C=CHUNK,
                  hidden_dim=6912, num_blocks=26,
                  rope_local=ROPE_LOCAL, rope_global=ROPE_GLOBAL)

    if arch == "lora":
        from trainCPTAC_GEMMA3_1K_lora_scratch import apply_lora_to_model, LoRALinear
        base = apply_lora_to_model(base, r=8, alpha=16)
        model = AGIV2GForCausalLMT(base, use_gc=False)

    elif arch == "fftrope":
        from trainCPTAC_AGI_GEMMA3_1K_fftrope_scratch import (
            SincCausalRoPE, make_phase_locked_decoder_forward)
        fft = SincCausalRoPE(num_layers=26, num_heads=4, head_dim=256, target_params=745472)
        for idx, block in enumerate(base.blocks):
            block.forward = types.MethodType(
                make_phase_locked_decoder_forward(idx, fft), block)
        model = AGIV2GForCausalLMT(base, use_gc=False)
        model.fft_phase_lock = fft

    elif arch == "lorafftrope":
        from trainCPTAC_AGI_GEMMA3_1K_lorafftrope_scratch import (
            apply_lora_to_model, SincCausalRoPE, make_phase_locked_decoder_forward)
        base = apply_lora_to_model(base, r=4, alpha=8)
        fft = SincCausalRoPE(num_layers=26, num_heads=4, head_dim=256, target_params=372736)
        for idx, block in enumerate(base.blocks):
            block.forward = types.MethodType(
                make_phase_locked_decoder_forward(idx, fft), block)
        model = AGIV2GForCausalLMT(base, use_gc=False)
        model.fft_phase_lock = fft

    elif arch == "fra":
        from trainCPTAC_AGI_GEMMA3_1K_fra_scratch import (
            SincCausalRoPE, make_fra_decoder_forward)
        fft = SincCausalRoPE(num_layers=26, num_heads=4, head_dim=256, target_params=745472)
        for idx, block in enumerate(base.blocks):
            block.forward = types.MethodType(
                make_fra_decoder_forward(idx, fft), block)
        model = AGIV2GForCausalLMT(base, use_gc=False)
        model.fft_phase_lock = fft

    elif arch == "lorafra":
        from trainCPTAC_AGI_GEMMA3_1K_lorafra_scratch import (
            apply_lora_to_model, SincCausalRoPE, make_fra_decoder_forward)
        base = apply_lora_to_model(base, r=4, alpha=8)
        fft = SincCausalRoPE(num_layers=26, num_heads=4, head_dim=256, target_params=372736)
        for idx, block in enumerate(base.blocks):
            block.forward = types.MethodType(
                make_fra_decoder_forward(idx, fft), block)
        model = AGIV2GForCausalLMT(base, use_gc=False)
        model.fft_phase_lock = fft
    elif arch == "duality":
        from trainCPTAC_AGI_GEMMA3_1K_duality_scratch import (
            SincCausalRoPE, make_dual_fusion_decoder_forward)
        fft = SincCausalRoPE(num_layers=26, num_heads=4, head_dim=256, target_params=745472)
        for idx, block in enumerate(base.blocks):
            block.forward = types.MethodType(
                make_dual_fusion_decoder_forward(idx, fft), block)
        model = AGIV2GForCausalLMT(base, use_gc=False)
        model.fft_phase_lock = fft
    elif arch == "loraduality":
        from trainCPTAC_AGI_GEMMA3_1K_loraduality_scratch import (
            apply_lora_to_model, SincCausalRoPE, make_dual_fusion_decoder_forward)
        base = apply_lora_to_model(base, r=4, alpha=8)
        fft = SincCausalRoPE(num_layers=26, num_heads=4, head_dim=256, target_params=372736)
        for idx, block in enumerate(base.blocks):
            block.forward = types.MethodType(
                make_dual_fusion_decoder_forward(idx, fft), block)
        model = AGIV2GForCausalLMT(base, use_gc=False)
        model.fft_phase_lock = fft
    elif arch == "headdual":
        from trainCPTAC_AGI_GEMMA3_1K_headdual_scratch import (
            SincCausalRoPE, make_head_decoupled_decoder_forward)
        fft = SincCausalRoPE(num_layers=26, target_params=745472)
        for idx, block in enumerate(base.blocks):
            block.forward = types.MethodType(
                make_head_decoupled_decoder_forward(idx, fft), block)
        model = AGIV2GForCausalLMT(base, use_gc=False)
        model.fft_phase_lock = fft
    elif arch == "loraheaddual":
        from trainCPTAC_AGI_GEMMA3_1K_loraheaddual_scratch import (
            apply_lora_to_model, SincCausalRoPE, make_head_decoupled_decoder_forward)
        base = apply_lora_to_model(base, r=4, alpha=8)
        fft = SincCausalRoPE(num_layers=26, target_params=372736)
        for idx, block in enumerate(base.blocks):
            block.forward = types.MethodType(
                make_head_decoupled_decoder_forward(idx, fft), block)
        model = AGIV2GForCausalLMT(base, use_gc=False)
        model.fft_phase_lock = fft
    elif arch == "gateheaddual":
        from trainCPTAC_AGI_GEMMA3_1K_gateheaddual_scratch import (
            GatedSincCausalRoPE, make_gated_head_decoupled_decoder_forward)
        fft = GatedSincCausalRoPE(num_layers=26, target_params=745472)
        for idx, block in enumerate(base.blocks):
            block.forward = types.MethodType(
                make_gated_head_decoupled_decoder_forward(idx, fft), block)
        model = AGIV2GForCausalLMT(base, use_gc=False)
        model.fft_phase_lock = fft
    else:
        raise ValueError(f"未知架構: {arch}")

    print(f"  📂 載入 checkpoint: {ckpt_path}")
    sd = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(sd, strict=False)
    model = model.eval().to(DEVICE).to(torch.bfloat16)
    return model

# ──────────────────────────────────────────
# Dim 1: 注意力熵
# ──────────────────────────────────────────
def eval_attention_entropy(model, batch):
    B, L = batch.shape
    entropies = []
    with torch.no_grad():
        bm = model.base_model
        X = bm.embedding(batch)
        for layer_idx, block in enumerate(bm.blocks):
            normed_X = block.input_layernorm(X)
            Q = block.W_q_loc(normed_X).view(B, L, block.num_heads, block.head_dim)
            K = block.W_k_loc(normed_X).view(B, L, block.num_kv_heads, block.head_dim)
            Q = block.q_norm(Q).transpose(1,2).float()
            K = block.k_norm(K).transpose(1,2).float()
            scores = torch.matmul(Q, K.transpose(-2,-1)) / math.sqrt(block.head_dim)
            mask = torch.triu(torch.full((L,L), float('-inf'), device=scores.device), diagonal=1)
            scores = scores + mask
            w = torch.softmax(scores, dim=-1)
            w = torch.nan_to_num(w, nan=0.0)
            H = -(w * torch.log(w + 1e-12)).sum(-1).mean().item()
            entropies.append(H)
            X = block(X)
    return float(np.mean(entropies))

# ──────────────────────────────────────────
# Dim 2: 表示空間各向同性
# ──────────────────────────────────────────
def eval_isotropy(model, batch):
    with torch.no_grad():
        bm = model.base_model
        X = bm.embedding(batch)
        for block in bm.blocks:
            X = block(X)
        feat = X.view(-1, X.size(-1)).float()
        feat = feat - feat.mean(0, keepdim=True)
        U, S, V = torch.linalg.svd(feat, full_matrices=False)
        s = S.cpu().numpy()
        s_norm = s / (s.sum() + 1e-12)
        sv_ent = -np.sum(s_norm * np.log(s_norm + 1e-12))
        isotropy = sv_ent / np.log(len(s))
    return float(isotropy)

# ──────────────────────────────────────────
# Dim 3: 長度外推 / Needle-in-Haystack
# ──────────────────────────────────────────
def eval_length_extrapolation(model, lengths=[128, 256, 512, 1024]):
    results = {}
    for L in lengths:
        if L > CHUNK * 2:   # 超過模型訓練窗口時跳過，避免 OOM
            results[L] = None
            continue
        torch.manual_seed(42)
        inputs = torch.randint(0, 262144, (1, L)).to(DEVICE)
        needle = L // 2
        inputs[0, needle:needle+4] = torch.tensor([1000,2000,3000,4000], device=DEVICE)
        inputs[0, -4:] = torch.tensor([1000,2000,3000,4000], device=DEVICE)
        with torch.no_grad():
            bm = model.base_model
            X = bm.embedding(inputs)
            last = bm.blocks[-1]
            normed_X = last.input_layernorm(X)
            Q = last.W_q_loc(normed_X).view(1, L, last.num_heads, last.head_dim)
            K = last.W_k_loc(normed_X).view(1, L, last.num_kv_heads, last.head_dim)
            Q = last.q_norm(Q).transpose(1,2).float()
            K = last.k_norm(K).transpose(1,2).float()
            scores = torch.matmul(Q, K.transpose(-2,-1)) / math.sqrt(last.head_dim)
            w = torch.softmax(scores, dim=-1).mean(1)   # (1,L,L)
            needle_attn = w[0, -4:, needle:needle+4].sum().item()
            distract_attn = w[0, -4:, :needle].mean().item()
            results[L] = needle_attn / (distract_attn + 1e-12)
    return results

# ──────────────────────────────────────────
# Dim 4: 干擾魯棒性
# ──────────────────────────────────────────
def eval_distractor_robustness(model, base_len=128, num_distractors=[0,1,2,4]):
    scores = {}
    for nd in num_distractors:
        total = min(base_len + nd * 64, CHUNK)
        torch.manual_seed(123)
        inp = torch.randint(0, 262144, (1, total)).to(DEVICE)
        inp[0, :4] = torch.tensor([5000,5001,5002,5003], device=DEVICE)
        inp[0, -4:] = torch.tensor([5000,5001,5002,5003], device=DEVICE)
        with torch.no_grad():
            bm = model.base_model
            X = bm.embedding(inp)
            mid = bm.blocks[13]
            normed_X = mid.input_layernorm(X)
            Q = mid.W_q_loc(normed_X).view(1, total, mid.num_heads, mid.head_dim)
            K = mid.W_k_loc(normed_X).view(1, total, mid.num_kv_heads, mid.head_dim)
            Q = mid.q_norm(Q).transpose(1,2).float()
            K = mid.k_norm(K).transpose(1,2).float()
            sc = torch.matmul(Q, K.transpose(-2,-1)) / math.sqrt(mid.head_dim)
            w = torch.softmax(sc, dim=-1).mean(1)
            scores[nd] = float(w[0, -4:, :4].sum().item())
    return scores

# ──────────────────────────────────────────
# Dim 5: 週期性結構捕捉
# ──────────────────────────────────────────
def eval_periodic_capture(model, seq_len=CHUNK, period=8):
    torch.manual_seed(999)
    def make_periodic():
        s = torch.randint(0, 262144, (1, seq_len)).to(DEVICE)
        for i in range(0, seq_len-4, period):
            s[0, i:i+4] = torch.tensor([888,999,888,999], device=DEVICE)
        return s
    seq_p = make_periodic()
    seq_r = torch.randint(0, 262144, (1, seq_len)).to(DEVICE)

    def spectral_entropy(inputs):
        with torch.no_grad():
            bm = model.base_model
            X = bm.embedding(inputs)
            for block in bm.blocks:
                X = block(X)
            feat = X[0].float().cpu().numpy()
            fft_amp = np.abs(np.fft.rfft(feat, axis=0))
            p = fft_amp / (fft_amp.sum(0, keepdims=True) + 1e-12)
            return float(np.mean(-np.sum(p * np.log(p + 1e-12), axis=0)))

    ep = spectral_entropy(seq_p)
    er = spectral_entropy(seq_r)
    return {"periodic": ep, "random": er, "PCI": er / (ep + 1e-12)}

# ──────────────────────────────────────────
# 從 CSV 讀最佳 eval loss
# ──────────────────────────────────────────
def best_eval_from_log(log_path):
    path = os.path.expanduser(log_path)
    best = float('inf')
    latest = 0
    if not os.path.exists(path): return None, None
    with open(path) as f:
        for row in csv.reader(f):
            try:
                step = int(row[0])
                if step > latest: latest = step
                if row[3].strip():
                    v = float(row[3])
                    if v < best: best = v
            except: pass
    return best if best < float('inf') else None, latest

# ──────────────────────────────────────────
# 主程式
# ──────────────────────────────────────────
MODELS = [
    ("LoRA",         "lora",        "~/gemma3_scratch_1k_lora_log.csv",
                                   "~/gemma3_scratch_checkpoints_1K_lora/best_cpt_model.pth"),
    ("FFT-RoPE",     "fftrope",     "~/agigemma3_scratch_1k_fftrope_log.csv",
                                   "~/agigemma3_scratch_checkpoints_1K_fftrope/best_cpt_model.pth"),
    ("LoRA+FFT",     "lorafftrope", "~/agigemma3_scratch_1k_lorafftrope_log.csv",
                                   "~/agigemma3_scratch_checkpoints_1K_lorafftrope/best_cpt_model.pth"),
    ("Pure FRA",     "fra",         "~/agigemma3_scratch_1k_fra_log.csv",
                                   "~/agigemma3_scratch_checkpoints_1K_fra/best_cpt_model.pth"),
    ("LoRA+FRA",     "lorafra",     "~/agigemma3_scratch_1k_lorafra_log.csv",
                                   "~/agigemma3_scratch_checkpoints_1K_lorafra/best_cpt_model.pth"),
    ("Duality",      "duality",     "~/agigemma3_scratch_1k_duality_log.csv",
                                   "~/agigemma3_scratch_checkpoints_1K_duality/best_cpt_model.pth"),
    ("LoRA+Duality", "loraduality", "~/agigemma3_scratch_1k_loraduality_log.csv",
                                   "~/agigemma3_scratch_checkpoints_1K_loraduality/best_cpt_model.pth"),
    ("HeadDual",     "headdual",    "~/agigemma3_scratch_1k_headdual_log.csv",
                                   "~/agigemma3_scratch_checkpoints_1K_headdual/best_cpt_model.pth"),
    ("LoRA+HeadDual","loraheaddual","~/agigemma3_scratch_1k_loraheaddual_log.csv",
                                   "~/agigemma3_scratch_checkpoints_1K_loraheaddual/best_cpt_model.pth"),
    ("gateHeadDual", "gateheaddual","~/agigemma3_scratch_1k_gateheaddual_log.csv",
                                   "~/agigemma3_scratch_checkpoints_1K_gateheaddual/best_cpt_model.pth"),
]

def main():
    torch.manual_seed(2026)
    batch = torch.randint(0, 262144, (1, CHUNK)).to(DEVICE)

    all_results = {}
    for label, arch, log_path, ckpt_rel in MODELS:
        ckpt_path = os.path.expanduser(ckpt_rel)
        best_loss, latest_step = best_eval_from_log(log_path)
        best_loss_str = f"{best_loss:.4f}" if best_loss is not None else "N/A"
        print(f"\n{'='*65}")
        print(f"🧬 評估: {label} | 最新步數: {latest_step} | 最佳EvalLoss: {best_loss_str}")
        print(f"{'='*65}")

        if not os.path.exists(ckpt_path):
            print(f"  ❌ Checkpoint 不存在，跳過")
            continue

        model = build_model(arch, ckpt_path)

        print("  ⚡ Dim1: 注意力熵...")
        d1 = eval_attention_entropy(model, batch)
        print("  ⚡ Dim2: 表示空間各向同性...")
        d2 = eval_isotropy(model, batch)
        print("  ⚡ Dim3: 長度外推...")
        d3 = eval_length_extrapolation(model, [128, 256, 512])
        print("  ⚡ Dim4: 干擾魯棒性...")
        d4 = eval_distractor_robustness(model)
        print("  ⚡ Dim5: 週期結構捕捉...")
        d5 = eval_periodic_capture(model)

        all_results[label] = {
            "best_eval_loss":   best_loss,
            "latest_step":      latest_step,
            "attn_entropy":     d1,
            "isotropy":         d2,
            "extrapolation":    d3,
            "distractor_rob":   d4,
            "periodic_capture": d5,
        }

        del model
        torch.cuda.empty_cache()

    # ── 輸出比較表 ──
    print("\n\n" + "="*90)
    print("📊 SKILL EVALUATION REPORT — From-Scratch 5-Way Comparison")
    print("="*90)
    hdr = f"{'模型':12s} | {'BestEvalLoss':>12} | {'D1:AttnEntropy':>14} | {'D2:Isotropy':>11} | {'D3:Extrap@256':>13} | {'D4:Rob(4)':>9} | {'D5:PCI':>8}"
    print(hdr)
    print("-"*90)
    for label, r in all_results.items():
        d3_256 = r['extrapolation'].get(256, None)
        d3_str = f"{d3_256:.4f}" if d3_256 is not None else "  N/A  "
        d4_str = f"{r['distractor_rob'].get(4, 0.0):.4f}"
        best_loss_val = r['best_eval_loss']
        best_loss_str = f"{best_loss_val:.4f}" if best_loss_val is not None else "    N/A     "
        print(f"{label:12s} | {best_loss_str:>12s} | {r['attn_entropy']:>14.4f} | {r['isotropy']:>11.4f} | {d3_str:>13} | {d4_str:>9} | {r['periodic_capture']['PCI']:>8.4f}")
    print("="*90)

    # 儲存 JSON
    out_path = os.path.expanduser("~/skill_eval_scratch_5way_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n💾 詳細結果已存至: {out_path}")

if __name__ == "__main__":
    main()
