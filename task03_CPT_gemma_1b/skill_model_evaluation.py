import os
import sys
import math
import time
import json
import torch
import torch.nn as nn
import numpy as np

# Append parent and local paths for custom imports
sys.path.append("/mnt/home/caijingnina-gmai-11e66a/wisdomCoreV2")
sys.path.append("/mnt/home/caijingnina-gmai-11e66a/wisdomCoreV2/task03_CPT_gemma_1b")

from GEMMA3 import GEMMA3
from utils import AGIV2GForCausalLMT, transplant_and_freeze
from trainCPTAC_AGI_GEMMA3_1K_lorafftrope import FFTPhaseLockRoPE, make_phase_locked_decoder_forward
from trainCPTAC_GEMMA3_1K_lora import apply_lora_to_model

# -------------------------------------------------------------
# [ SKILL CONFIGURATION & INITIALIZERS ]
# -------------------------------------------------------------
MODEL_ID = "google/gemma-3-1b-it"
CHUNK = 256
ROPE_LOCAL = 128
ROPE_GLOBAL = 512

class ModelEvaluator:
    def __init__(self, model_type, checkpoint_path=None, device="cuda"):
        self.model_type = model_type
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.model = self.load_model()
        
    def load_model(self):
        print(f"\n🔄 [Initializer] Loading model configuration for: {self.model_type}")
        base = GEMMA3(vocab_size=262144, D=1152, C=CHUNK, hidden_dim=6912, num_blocks=26, rope_local=ROPE_LOCAL, rope_global=ROPE_GLOBAL)
        base = transplant_and_freeze(MODEL_ID, base)
        
        if self.model_type == "lora":
            # Pure LoRA: rank r=8, alpha=16
            base = apply_lora_to_model(base, r=8, alpha=16)
            model = AGIV2GForCausalLMT(base, use_gc=True)
            
        elif self.model_type == "fftrope":
            # Pure FFT-RoPE: target_params=745,472
            fft_phase_lock = FFTPhaseLockRoPE(num_layers=26, num_heads=4, head_dim=256, target_params=745472)
            for idx, block in enumerate(base.blocks):
                block.forward = make_phase_locked_decoder_forward(idx, fft_phase_lock).__get__(block, block.__class__)
            model = AGIV2GForCausalLMT(base, use_gc=True)
            model.fft_phase_lock = fft_phase_lock
            
        elif self.model_type == "hybrid":
            # Hybrid: LoRA r=4 (372k params) + FFT-RoPE (372k params)
            base = apply_lora_to_model(base, r=4, alpha=8)
            fft_phase_lock = FFTPhaseLockRoPE(num_layers=26, num_heads=4, head_dim=256, target_params=372736)
            for idx, block in enumerate(base.blocks):
                block.forward = make_phase_locked_decoder_forward(idx, fft_phase_lock).__get__(block, block.__class__)
            model = AGIV2GForCausalLMT(base, use_gc=True)
            model.fft_phase_lock = fft_phase_lock
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
            
        if self.checkpoint_path and os.path.exists(self.checkpoint_path):
            print(f"📂 [Weights] Loading checkpoint state dict from: {self.checkpoint_path}")
            model.load_state_dict(torch.load(self.checkpoint_path, map_location="cpu"), strict=False)
        else:
            print(f"⚠️ [Weights] Checkpoint not found at {self.checkpoint_path}. Using zero-init base model weights.")
            
        return model.eval().to(self.device)

    # -------------------------------------------------------------
    # [ DIMENSION 1: ATTENTION ENTROPY ANALYSIS ]
    # -------------------------------------------------------------
    def evaluate_attention_entropy(self, batch_inputs):
        """Measures Shannon Entropy across all attention layers to evaluate focusing sharpness."""
        entropies = []
        B, L = batch_inputs.shape
        eps = 1e-12
        
        with torch.no_grad():
            base_model = self.model.base_model
            X = base_model.embed(batch_inputs)
            
            for layer_idx, block in enumerate(base_model.blocks):
                normed_X = block.input_layernorm(X)
                
                # Extract linear states
                Q = block.W_q_loc(normed_X).view(B, L, block.num_heads, block.head_dim)
                K = block.W_k_loc(normed_X).view(B, L, block.num_kv_heads, block.head_dim)
                
                Q = block.q_norm(Q)
                K = block.k_norm(K)
                
                # Apply phase locking if FFT module is present
                delta_theta = self.model.fft_phase_lock(normed_X, layer_idx) if hasattr(self.model, "fft_phase_lock") else None
                
                from trainCPTAC_AGI_GEMMA3_1K_lorafftrope import apply_phase_locked_rope
                Q = apply_phase_locked_rope(Q, block.head_dim, base_freq=block.rope_base, delta_theta=delta_theta)
                
                delta_theta_k = delta_theta.mean(dim=2, keepdim=True) if delta_theta is not None else None
                K = apply_phase_locked_rope(K, block.head_dim, base_freq=block.rope_base, delta_theta=delta_theta_k)
                
                # Compute scores
                Q = Q.transpose(1, 2)
                K = K.transpose(1, 2)
                scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(block.head_dim)
                
                mask = torch.triu(torch.full((L, L), float('-inf'), device=scores.device), diagonal=1)
                scores = scores + mask
                
                attn_weights = torch.softmax(scores, dim=-1)
                attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
                
                # Shannon Entropy: H(A) = -sum(p * log(p))
                entropy = -torch.sum(attn_weights * torch.log(attn_weights + eps), dim=-1)
                mean_entropy = entropy.mean().item()
                entropies.append(mean_entropy)
                
                # Forward prop
                X = block(X)
                
        return {
            "mean_attention_entropy": float(np.mean(entropies)),
            "layer_wise_entropies": [float(e) for e in entropies]
        }

    # -------------------------------------------------------------
    # [ DIMENSION 2: REPRESENTATION ISOTROPY ]
    # -------------------------------------------------------------
    def evaluate_representation_isotropy(self, batch_inputs):
        """Performs Singular Value Decomposition (SVD) on final layer hidden states to detect dimensional collapse."""
        with torch.no_grad():
            base_model = self.model.base_model
            X = base_model.embed(batch_inputs)
            for block in base_model.blocks:
                X = block(X)
                
            features = X.view(-1, X.size(-1)) # (B * L, D)
            features_centered = features - features.mean(dim=0, keepdim=True)
            
            # SVD Decomposition
            U, S, V = torch.linalg.svd(features_centered.float(), full_matrices=False)
            singular_values = S.cpu().numpy()
            
            # Normalized Singular Value Entropy (Isotropy Score)
            s_norm = singular_values / (np.sum(singular_values) + 1e-12)
            sv_entropy = -np.sum(s_norm * np.log(s_norm + 1e-12))
            max_entropy = np.log(len(singular_values))
            isotropy_score = sv_entropy / max_entropy
            
            # Cosine similarity variance (Narrow cone check)
            normed_features = features / (torch.norm(features, dim=-1, keepdim=True) + 1e-12)
            rand_indices_1 = torch.randint(0, len(normed_features), (500,))
            rand_indices_2 = torch.randint(0, len(normed_features), (500,))
            cos_sims = torch.sum(normed_features[rand_indices_1] * normed_features[rand_indices_2], dim=-1)
            mean_cos_sim = cos_sims.mean().item()
            
        return {
            "isotropy_score": float(isotropy_score),
            "mean_cosine_similarity": float(mean_cos_sim),
            "top_singular_values": [float(val) for val in singular_values[:5]]
        }

    # -------------------------------------------------------------
    # [ DIMENSION 3: LENGTH EXTRAPOLATION BENCHMARK ]
    # -------------------------------------------------------------
    def evaluate_length_extrapolation(self, target_lengths=[128, 512, 1024, 2048, 4096]):
        """Benchmarks the model's geometric retrieval focus (Needle-in-a-Haystack simulation) across context lengths."""
        length_scores = {}
        for length in target_lengths:
            # Construct synthetic long prompt with a highly correlated token match placed in the middle
            torch.manual_seed(42)
            inputs = torch.randint(0, 262144, (1, length)).to(self.device)
            needle_pos = length // 2
            
            # Inject key pattern in the middle (Needle) and a query at the end
            inputs[0, needle_pos:needle_pos+4] = torch.tensor([1000, 2000, 3000, 4000], device=self.device)
            inputs[0, -4:] = torch.tensor([1000, 2000, 3000, 4000], device=self.device)
            
            with torch.no_grad():
                base_model = self.model.base_model
                X = base_model.embed(inputs)
                
                # Measure final attention layer alignment to the target needle position
                # Hooking attention weights of the last layer blocks
                last_block = base_model.blocks[-1]
                normed_X = last_block.input_layernorm(X)
                
                Q = last_block.W_q_loc(normed_X).view(1, length, last_block.num_heads, last_block.head_dim)
                K = last_block.W_k_loc(normed_X).view(1, length, last_block.num_kv_heads, last_block.head_dim)
                
                Q = last_block.q_norm(Q)
                K = last_block.k_norm(K)
                
                delta_theta = self.model.fft_phase_lock(normed_X, 25) if hasattr(self.model, "fft_phase_lock") else None
                
                from trainCPTAC_AGI_GEMMA3_1K_lorafftrope import apply_phase_locked_rope
                Q = apply_phase_locked_rope(Q, last_block.head_dim, base_freq=last_block.rope_base, delta_theta=delta_theta)
                
                delta_theta_k = delta_theta.mean(dim=2, keepdim=True) if delta_theta is not None else None
                K = apply_phase_locked_rope(K, last_block.head_dim, base_freq=last_block.rope_base, delta_theta=delta_theta_k)
                
                Q = Q.transpose(1, 2)
                K = K.transpose(1, 2)
                scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(last_block.head_dim)
                attn_weights = torch.softmax(scores, dim=-1).mean(dim=1) # Average across heads -> (1, L, L)
                
                # Measure how much attention the final query tokens (last 4 tokens) pay to the needle tokens
                query_attn_to_needle = attn_weights[0, -4:, needle_pos:needle_pos+4].sum().item()
                # Normalize against attention to random distraction blocks
                distractor_attn = attn_weights[0, -4:, :needle_pos].mean().item()
                
                retrieval_ratio = query_attn_to_needle / (distractor_attn + 1e-12)
                length_scores[length] = {
                    "needle_attention": float(query_attn_to_needle),
                    "distractor_mean_attention": float(distractor_attn),
                    "retrieval_ratio": float(retrieval_ratio)
                }
                
        return length_scores

    # -------------------------------------------------------------
    # [ DIMENSION 4: IN-CONTEXT DISTRACTOR ROBUSTNESS ]
    # -------------------------------------------------------------
    def evaluate_distractor_robustness(self, base_length=256, num_distractors=[0, 1, 2, 4]):
        """Evaluates model retention of context relationships when irrelevant distractors are injected."""
        robustness_scores = {}
        for num in num_distractors:
            # Dynamic prompt construction
            total_len = base_length + (num * 128)
            torch.manual_seed(123)
            inputs = torch.randint(0, 262144, (1, total_len)).to(self.device)
            
            # Place base key at sequence start
            inputs[0, :4] = torch.tensor([5000, 5001, 5002, 5003], device=self.device)
            # Place base query at sequence end
            inputs[0, -4:] = torch.tensor([5000, 5001, 5002, 5003], device=self.device)
            
            with torch.no_grad():
                base_model = self.model.base_model
                X = base_model.embed(inputs)
                
                # Check mid-layer blocks (where context filtering is strongest)
                mid_block = base_model.blocks[13]
                normed_X = mid_block.input_layernorm(X)
                
                Q = mid_block.W_q_loc(normed_X).view(1, total_len, mid_block.num_heads, mid_block.head_dim)
                K = mid_block.W_k_loc(normed_X).view(1, total_len, mid_block.num_kv_heads, mid_block.head_dim)
                
                Q = mid_block.q_norm(Q)
                K = mid_block.k_norm(K)
                
                delta_theta = self.model.fft_phase_lock(normed_X, 13) if hasattr(self.model, "fft_phase_lock") else None
                
                from trainCPTAC_AGI_GEMMA3_1K_lorafftrope import apply_phase_locked_rope
                Q = apply_phase_locked_rope(Q, mid_block.head_dim, base_freq=mid_block.rope_base, delta_theta=delta_theta)
                
                delta_theta_k = delta_theta.mean(dim=2, keepdim=True) if delta_theta is not None else None
                K = apply_phase_locked_rope(K, mid_block.head_dim, base_freq=mid_block.rope_base, delta_theta=delta_theta_k)
                
                Q = Q.transpose(1, 2)
                K = K.transpose(1, 2)
                scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(mid_block.head_dim)
                attn_weights = torch.softmax(scores, dim=-1).mean(dim=1)
                
                # Focus retention: sum of attention from final query to initial base key
                focus_retention = attn_weights[0, -4:, :4].sum().item()
                robustness_scores[num] = float(focus_retention)
                
        return robustness_scores

    # -------------------------------------------------------------
    # [ DIMENSION 5: PERIODIC MOTIF CAPTURE (GENOMICS/CPTAC) ]
    # -------------------------------------------------------------
    def evaluate_periodic_motif_capture(self, seq_len=256, period=8):
        """Evaluates frequency-domain capture of structural periodicity."""
        torch.manual_seed(999)
        
        # Scenario A: Periodic Input Sequence (Motifs repeat every `period` tokens)
        seq_periodic = torch.randint(0, 262144, (1, seq_len)).to(self.device)
        for i in range(0, seq_len - 4, period):
            seq_periodic[0, i:i+4] = torch.tensor([888, 999, 888, 999], device=self.device)
            
        # Scenario B: Chaotic/Random Input Sequence
        seq_random = torch.randint(0, 262144, (1, seq_len)).to(self.device)
        
        def compute_harmonic_entropy(inputs):
            with torch.no_grad():
                base_model = self.model.base_model
                X = base_model.embed(inputs)
                
                # Pull final hidden layer representations
                for block in base_model.blocks:
                    X = block(X)
                
                # FFT analysis across sequence features to see if periodic signal yields structured spectral peaks
                features = X[0].float().cpu().numpy() # (L, D)
                features_fft = np.abs(np.fft.rfft(features, axis=0)) # (L//2 + 1, D)
                
                # Normalizing spectrum to obtain frequency power probability distribution
                power_distribution = features_fft / (np.sum(features_fft, axis=0, keepdims=True) + 1e-12)
                spectral_entropy = -np.sum(power_distribution * np.log(power_distribution + 1e-12), axis=0)
                
                # Standard Shannon Spectral Entropy: lower entropy = sharp periodic alignment
                return float(np.mean(spectral_entropy))
                
        entropy_periodic = compute_harmonic_entropy(seq_periodic)
        entropy_random = compute_harmonic_entropy(seq_random)
        
        # Periodic Capture Index (PCI): Ratio of structured periodicity response
        pci = entropy_random / (entropy_periodic + 1e-12)
        
        return {
            "spectral_entropy_periodic": entropy_periodic,
            "spectral_entropy_random": entropy_random,
            "periodic_capture_index": pci
        }

    # -------------------------------------------------------------
    # [ CORE PIPELINE DISPATCHER ]
    # -------------------------------------------------------------
    def run_full_pipeline(self):
        print("="*65)
        print(f"🧬 Starting Evaluation Pipeline for Model: {self.model_type.upper()}")
        print("="*65)
        
        # Create standard batch for general dimension metrics
        torch.manual_seed(2026)
        batch = torch.randint(0, 262144, (1, CHUNK)).to(self.device)
        
        results = {}
        
        # Dimension 1: Attention Entropy
        print("⚡ [Dimension 1] Calculating attention matrix Shannon entropy...")
        results["entropy_metrics"] = self.evaluate_attention_entropy(batch)
        
        # Dimension 2: Space Isotropy
        print("⚡ [Dimension 2] Checking hidden states space isotropy via SVD...")
        results["isotropy_metrics"] = self.evaluate_representation_isotropy(batch)
        
        # Dimension 3: Context Length Extrapolation
        print("⚡ [Dimension 3] Benchmarking context length extrapolation...")
        results["extrapolation_metrics"] = self.evaluate_length_extrapolation()
        
        # Dimension 4: Distractor Robustness
        print("⚡ [Dimension 4] Estimating context distractor robustness...")
        results["robustness_metrics"] = self.evaluate_distractor_robustness()
        
        # Dimension 5: Periodic Capture Index
        print("⚡ [Dimension 5] Analyzing periodic CPTAC genomic sequence capture...")
        results["periodic_metrics"] = self.evaluate_periodic_motif_capture()
        
        print(f"✅ Pipeline completed for {self.model_type}!")
        return results

# -------------------------------------------------------------
# [ SKILL CLI RUNNER ]
# -------------------------------------------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Skill Model Evaluation Pipeline")
    parser.add_argument("--model", type=str, default="hybrid", choices=["lora", "fftrope", "hybrid"], help="Model architecture type")
    parser.add_argument("--checkpoint", type=str, default=None, help="Absolute path to checkpoint .pth file")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path to save metrics")
    args = parser.parse_args()
    
    # Auto-resolve checkpoint defaults if not provided
    ckpt_map = {
        "lora": "~/gemma3_cpt_checkpoints_1K_lora/final_gemma3_cpt_1k_lora.pth",
        "fftrope": "~/agigemma3_cpt_checkpoints_1K_fftrope/final_agigemma3_cpt_1k_fftrope.pth",
        "hybrid": "~/agigemma3_cpt_checkpoints_1K_lorafftrope/final_agigemma3_cpt_1k_lorafftrope.pth"
    }
    
    checkpoint_path = args.checkpoint
    if not checkpoint_path:
        checkpoint_path = os.path.expanduser(ckpt_map[args.model])
        
    evaluator = ModelEvaluator(args.model, checkpoint_path=checkpoint_path)
    metrics = evaluator.run_full_pipeline()
    
    # Output report to CLI
    print("\n" + "="*80)
    print(f"📊 SKILL EVALUATION REPORT: {args.model.upper()}")
    print("="*80)
    print(f"1. Attention Entropy (Lower = Focused)   : {metrics['entropy_metrics']['mean_attention_entropy']:.6f}")
    print(f"2. Representation Isotropy (Higher = Rich): {metrics['isotropy_metrics']['isotropy_score']:.6f}")
    print(f"3. Extrapolation Retrieval @ 4096 tokens  : {metrics['extrapolation_metrics'].get(4096, {}).get('retrieval_ratio', 0.0):.6f}")
    print(f"4. Distractor Robustness (4 Distractors)   : {metrics['robustness_metrics'].get(4, 0.0):.6f}")
    print(f"5. Periodic Capture Index (Higher = Smart): {metrics['periodic_metrics']['periodic_capture_index']:.6f}")
    print("="*80)
    
    if args.output:
        with open(args.output, "w") as f:
            json.dump(metrics, f, indent=4)
        print(f"💾 Report saved successfully to: {args.output}")

if __name__ == "__main__":
    main()
