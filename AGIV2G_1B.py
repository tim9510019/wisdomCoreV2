import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ==========================================
# 基礎 SOTA 元件庫 (完美對齊 Gemma 3 1B 物理特性)
# ==========================================

class GemmaRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        norm_x = x.to(torch.float32)
        norm = torch.rsqrt(norm_x.pow(2).mean(-1, keepdim=True) + self.eps)
        return ((norm_x * norm).to(x.dtype)) * (1.0 + self.weight)

class GemmaFFN(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        gate = F.gelu(self.gate_proj(x), approximate="tanh")
        return self.down_proj(gate * self.up_proj(x))

def apply_rope(x, head_dim):
    """全局絕對位置編碼"""
    B, L, num_heads, D = x.shape
    position = torch.arange(L, device=x.device).unsqueeze(1).float()
    div_term = torch.exp(torch.arange(0, head_dim, 2, device=x.device).float() * -(math.log(1000000.0) / head_dim))
    freqs = position * div_term 
    emb = torch.cat((freqs, freqs), dim=-1).unsqueeze(0).unsqueeze(2) 
    
    sin_val = torch.sin(emb).to(x.dtype)
    cos_val = torch.cos(emb).to(x.dtype)
    
    x1 = x[..., : head_dim // 2]
    x2 = x[..., head_dim // 2 :]
    x_rotated = torch.cat((-x2, x1), dim=-1)
    
    return (x * cos_val) + (x_rotated * sin_val)

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    B, L, num_kv_heads, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, :, None, :].expand(B, L, num_kv_heads, n_rep, head_dim)
    return hidden_states.reshape(B, L, num_kv_heads * n_rep, head_dim)

# ==========================================
# 核心網路區塊 (1:3 稀疏架構分化)
# ==========================================

class AGIV2LocalBlock(nn.Module):
    # 🌟 已切換為 1B 級別優化預設值
    def __init__(self, D=1536, hidden_dim=6144, C=256, num_heads=6, num_kv_heads=2, head_dim=256):
        super().__init__()
        self.C = C
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_key_value_groups = num_heads // num_kv_heads
        
        self.input_layernorm = GemmaRMSNorm(D)
        self.post_attention_layernorm = GemmaRMSNorm(D)
        self.pre_feedforward_layernorm = GemmaRMSNorm(D)
        self.post_feedforward_layernorm = GemmaRMSNorm(D)
        
        self.q_norm = GemmaRMSNorm(head_dim)
        self.k_norm = GemmaRMSNorm(head_dim)
        
        self.W_q_loc = nn.Linear(D, num_heads * self.head_dim, bias=False)
        self.W_k_loc = nn.Linear(D, num_kv_heads * self.head_dim, bias=False)
        self.W_v_loc = nn.Linear(D, num_kv_heads * self.head_dim, bias=False)
        self.o_proj_loc = nn.Linear(num_heads * self.head_dim, D, bias=False)
        
        self.ffn = GemmaFFN(hidden_size=D, intermediate_size=hidden_dim)

    def forward(self, X, shift_size=0, n_split_index=None):
        B, L, D = X.shape
        device = X.device
        
        normed_X = self.input_layernorm(X)
        
        pad_len = (self.C - L % self.C) % self.C
        Z = F.pad(normed_X, (0, 0, 0, pad_len)) if pad_len > 0 else normed_X
        L_pad = L + pad_len
        N = L_pad // self.C
        
        Q = self.W_q_loc(Z).view(B, L_pad, self.num_heads, self.head_dim)
        K = self.W_k_loc(Z).view(B, L_pad, self.num_kv_heads, self.head_dim)
        V = self.W_v_loc(Z).view(B, L_pad, self.num_kv_heads, self.head_dim)
        
        Q = self.q_norm(Q)
        K = self.k_norm(K)
        
        Q = apply_rope(Q, self.head_dim)
        K = apply_rope(K, self.head_dim)
        
        K = repeat_kv(K, self.num_key_value_groups)
        V = repeat_kv(V, self.num_key_value_groups)
        
        if shift_size > 0:
            Q = torch.roll(Q, shifts=-shift_size, dims=1)
            K = torch.roll(K, shifts=-shift_size, dims=1)
            V = torch.roll(V, shifts=-shift_size, dims=1)
            
        Q_c = Q.view(B, N, self.C, self.num_heads, self.head_dim).transpose(2, 3) 
        K_c = K.view(B, N, self.C, self.num_heads, self.head_dim).transpose(2, 3)
        V_c = V.view(B, N, self.C, self.num_heads, self.head_dim).transpose(2, 3)
        
        scale_factor = math.sqrt(math.sqrt(self.head_dim) * 50.0)
        Q_c_scaled = Q_c / scale_factor
        K_c_scaled = K_c / scale_factor
        
        attn_weights = torch.matmul(Q_c_scaled, K_c_scaled.transpose(-1, -2))
        attn_weights = torch.tanh(attn_weights) * 50.0 
        
        causal_mask = torch.tril(torch.ones((self.C, self.C), device=device, dtype=torch.bool))
        
        if shift_size > 0:
            region_idx = torch.zeros(L_pad, device=device)
            region_idx[-shift_size:] = 1 
            mask_windows = region_idx.view(1, N, 1, self.C, 1)
            shift_mask = (mask_windows == mask_windows.transpose(-1, -2)) 
            sdpa_mask = shift_mask & causal_mask.view(1, 1, 1, self.C, self.C)
        else:
            sdpa_mask = causal_mask.view(1, 1, 1, self.C, self.C)
            
        attn_weights = attn_weights.masked_fill(~sdpa_mask, torch.finfo(attn_weights.dtype).min)
            
        attn_probs = F.softmax(attn_weights, dim=-1)
        
        Z_hat_chunked = torch.matmul(attn_probs, V_c) 
        Z_hat_chunked = Z_hat_chunked.transpose(2, 3).reshape(B, L_pad, self.num_heads * self.head_dim)
        
        Z_hat_shifted = self.o_proj_loc(Z_hat_chunked)
        Z_hat = torch.roll(Z_hat_shifted, shifts=shift_size, dims=1) if shift_size > 0 else Z_hat_shifted
        
        if pad_len > 0:
            Z_hat = Z_hat[:, :L, :]
            
        X_res1 = X + self.post_attention_layernorm(Z_hat)
        
        normed_X_ffn = self.pre_feedforward_layernorm(X_res1)
        ffn_out = self.ffn(normed_X_ffn)
        
        Output = X_res1 + self.post_feedforward_layernorm(ffn_out)
        return Output


class AGIV2GlobalBlock(nn.Module):
    # 🌟 已切換為 1B 級別優化預設值
    def __init__(self, D=1536, hidden_dim=6144, K=256, M=256, C=256, num_heads=6, num_kv_heads=2, head_dim=256):
        super().__init__()
        self.D = D
        self.C = C
        self.M = M
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_key_value_groups = num_heads // num_kv_heads
        
        self.input_layernorm = GemmaRMSNorm(D)
        self.post_attention_layernorm = GemmaRMSNorm(D)
        self.pre_feedforward_layernorm = GemmaRMSNorm(D)
        self.post_feedforward_layernorm = GemmaRMSNorm(D)
        
        self.fft_norm = GemmaRMSNorm(D)
        self.mem_norm = GemmaRMSNorm(D)
        
        self.q_norm = GemmaRMSNorm(head_dim)
        self.k_norm = GemmaRMSNorm(head_dim)
        
        self.gate_fft = nn.Parameter(torch.zeros(1)) # 建議從 0 開始，讓殘差穩定
        self.gate_mem = nn.Parameter(torch.zeros(1)) # 建議從 0 開始，讓殘差穩定
        
        self.omegas = nn.Parameter(torch.randn(K))
        self.mlp_H = nn.Sequential(
            nn.Linear(2 * K, D),
            nn.GELU(),
            nn.Linear(D, D)
        )
        
        self.Q_mem = nn.Parameter(torch.randn(M, D))
        self.W_k_mem = nn.Linear(D, D, bias=False)
        self.W_v_mem = nn.Linear(D, D, bias=False)
        
        self.W_q_loc = nn.Linear(D, num_heads * self.head_dim, bias=False)
        self.W_k_loc = nn.Linear(D, num_kv_heads * self.head_dim, bias=False)
        self.W_v_loc = nn.Linear(D, num_kv_heads * self.head_dim, bias=False)
        self.o_proj_loc = nn.Linear(num_heads * self.head_dim, D, bias=False)
        
        self.W_q_cross = nn.Linear(D, num_heads * self.head_dim, bias=False)
        self.W_k_cross = nn.Linear(D, num_kv_heads * self.head_dim, bias=False)
        self.W_v_cross = nn.Linear(D, num_kv_heads * self.head_dim, bias=False)
        self.o_proj_cross = nn.Linear(num_heads * self.head_dim, D, bias=False)
        
        self.ffn = GemmaFFN(hidden_size=D, intermediate_size=hidden_dim)

    def forward(self, X, shift_size=0, n_split_index=None):
        B, L, D = X.shape
        device = X.device
        dtype = X.dtype
        
        if n_split_index is not None:
            seq_range = torch.arange(L, device=device).unsqueeze(0)
            past_mask = seq_range < n_split_index.unsqueeze(1)
            X_past = X * past_mask.unsqueeze(-1).to(dtype)
        else:
            X_past = X
            past_mask = None
            
        normed_X_fft = self.fft_norm(X_past)
        
        t = torch.arange(L, device=device, dtype=torch.float32)
        args = t.unsqueeze(1) * self.omegas.unsqueeze(0).to(torch.float32)
        gamma = torch.cat([torch.sin(args), torch.cos(args)], dim=-1).to(dtype)
        H = self.mlp_H(gamma)
        
        X_pad = F.pad(normed_X_fft, (0, 0, 0, L))  
        H_pad = F.pad(H, (0, 0, 0, L))  
        
        X_f = torch.fft.rfft(X_pad.to(torch.float32), dim=1)
        H_f = torch.fft.rfft(H_pad.to(torch.float32), dim=0).unsqueeze(0)

        Y_f = X_f * torch.conj(H_f)
        Y_sys1 = torch.fft.irfft(Y_f, n=2*L, dim=1).to(dtype)[:, :L, :]
        
        # 🚀 頻域縮放防護：穩定大模型在長序列下的浮點誤差
        Y_sys1 = Y_sys1 / math.sqrt(max(L, 1))
        
        X_res0 = X + (Y_sys1 * self.gate_fft.to(dtype))
        
        X_res0_past = X_res0 * past_mask.unsqueeze(-1).to(dtype) if past_mask is not None else X_res0
        normed_X_mem = self.mem_norm(X_res0_past)
        
        K_mem = self.W_k_mem(normed_X_mem)
        V_mem = self.W_v_mem(normed_X_mem)
        
        mem_pad = (self.C - L % self.C) % self.C
        K_mem_p = F.pad(K_mem, (0, 0, 0, mem_pad)) if mem_pad > 0 else K_mem
        V_mem_p = F.pad(V_mem, (0, 0, 0, mem_pad)) if mem_pad > 0 else V_mem
        L_mem_pad = K_mem_p.size(1)
        N_mem = L_mem_pad // self.C
        
        K_chunks = K_mem_p.view(B, N_mem, self.C, D)
        V_chunks = V_mem_p.view(B, N_mem, self.C, D)
        
        Q_exp = self.Q_mem.view(1, 1, self.M, D).expand(B, N_mem, -1, -1)
        
        if past_mask is not None:
            past_mask_pad = F.pad(past_mask, (0, mem_pad), value=False)
            past_mask_chunks = past_mask_pad.view(B, N_mem, self.C)
            
            chunk_valid = past_mask_chunks.any(dim=-1, keepdim=True)
            safe_past_mask = past_mask_chunks | ~chunk_valid 
            attn_mask = safe_past_mask.view(B * N_mem, 1, 1, self.C)
            
            updates = F.scaled_dot_product_attention(
                Q_exp.reshape(B * N_mem, 1, self.M, D),
                K_chunks.reshape(B * N_mem, 1, self.C, D),
                V_chunks.reshape(B * N_mem, 1, self.C, D),
                attn_mask=attn_mask,
                dropout_p=0.0, is_causal=False
            ).reshape(B, N_mem, self.M, D)
            
            chunk_valid_N = chunk_valid.view(B, N_mem, 1, 1)
            M_global_sum = (updates * chunk_valid_N).sum(dim=1, keepdim=True)
            valid_count = chunk_valid_N.sum(dim=1).view(B, 1, 1, 1).clamp(min=1)
            M_global_single = M_global_sum / valid_count
        else:
            updates = F.scaled_dot_product_attention(
                Q_exp.reshape(B * N_mem, 1, self.M, D),
                K_chunks.reshape(B * N_mem, 1, self.C, D),
                V_chunks.reshape(B * N_mem, 1, self.C, D),
                dropout_p=0.0, is_causal=False
            ).reshape(B, N_mem, self.M, D)
            M_global_single = updates.mean(dim=1, keepdim=True)
            
        M_global = M_global_single.expand(-1, N_mem, -1, -1)
        
        normed_X_loc = self.input_layernorm(X_res0)
        
        pad_len = mem_pad  
        Z = F.pad(normed_X_loc, (0, 0, 0, pad_len)) if pad_len > 0 else normed_X_loc
        L_pad = L_mem_pad
        N = N_mem
        
        Q_loc = self.W_q_loc(Z).view(B, L_pad, self.num_heads, self.head_dim)
        K_loc = self.W_k_loc(Z).view(B, L_pad, self.num_kv_heads, self.head_dim)
        V_loc = self.W_v_loc(Z).view(B, L_pad, self.num_kv_heads, self.head_dim)
        
        Q_loc = self.q_norm(Q_loc)
        K_loc = self.k_norm(K_loc)
        
        Q_loc = apply_rope(Q_loc, self.head_dim)
        K_loc = apply_rope(K_loc, self.head_dim)
        
        K_loc = repeat_kv(K_loc, self.num_key_value_groups)
        V_loc = repeat_kv(V_loc, self.num_key_value_groups)
        
        if shift_size > 0:
            Q_loc = torch.roll(Q_loc, shifts=-shift_size, dims=1)
            K_loc = torch.roll(K_loc, shifts=-shift_size, dims=1)
            V_loc = torch.roll(V_loc, shifts=-shift_size, dims=1)
            
        Q_c = Q_loc.view(B, N, self.C, self.num_heads, self.head_dim).transpose(2, 3)
        K_c = K_loc.view(B, N, self.C, self.num_heads, self.head_dim).transpose(2, 3)
        V_c = V_loc.view(B, N, self.C, self.num_heads, self.head_dim).transpose(2, 3)
        
        scale_factor = math.sqrt(math.sqrt(self.head_dim) * 50.0)
        Q_c_scaled = Q_c / scale_factor
        K_c_scaled = K_c / scale_factor
        
        attn_weights = torch.matmul(Q_c_scaled, K_c_scaled.transpose(-1, -2))
        attn_weights = torch.tanh(attn_weights) * 50.0
        
        causal_mask = torch.tril(torch.ones((self.C, self.C), device=device, dtype=torch.bool))
        
        if shift_size > 0:
            region_idx = torch.zeros(L_pad, device=device)
            region_idx[-shift_size:] = 1 
            mask_windows = region_idx.view(1, N, 1, self.C, 1)
            shift_mask = (mask_windows == mask_windows.transpose(-1, -2)) 
            sdpa_mask = shift_mask & causal_mask.view(1, 1, 1, self.C, self.C)
        else:
            sdpa_mask = causal_mask.view(1, 1, 1, self.C, self.C)
            
        attn_weights = attn_weights.masked_fill(~sdpa_mask, torch.finfo(attn_weights.dtype).min)
            
        attn_probs = F.softmax(attn_weights, dim=-1)
        
        Z_hat_chunked = torch.matmul(attn_probs, V_c)
        Z_hat_chunked = Z_hat_chunked.transpose(2, 3).reshape(B, L_pad, self.num_heads * self.head_dim)
        
        Z_hat_shifted = self.o_proj_loc(Z_hat_chunked)
        Z_hat = torch.roll(Z_hat_shifted, shifts=shift_size, dims=1) if shift_size > 0 else Z_hat_shifted
        
        Z_hat_chunks = Z_hat.view(B, N, self.C, -1)
        
        Q_cross = self.W_q_cross(Z_hat_chunks)
        Q_cross = Q_cross.view(B * N, self.C, self.num_heads, self.head_dim)
        
        K_cross = self.W_k_cross(M_global)
        K_cross = K_cross.view(B * N, self.M, self.num_kv_heads, self.head_dim)
        
        V_cross = self.W_v_cross(M_global)
        V_cross = V_cross.view(B * N, self.M, self.num_kv_heads, self.head_dim)
        
        K_cross = repeat_kv(K_cross, self.num_key_value_groups)
        V_cross = repeat_kv(V_cross, self.num_key_value_groups)
        
        Q_cross = Q_cross.transpose(1, 2)
        K_cross = K_cross.transpose(1, 2)
        V_cross = V_cross.transpose(1, 2)
        
        I_cross = F.scaled_dot_product_attention(
            Q_cross, K_cross, V_cross, dropout_p=0.0, is_causal=False 
        )
        
        I_cross = I_cross.transpose(1, 2).reshape(B, L_pad, self.num_heads * self.head_dim)
        I_cross = self.o_proj_cross(I_cross)
        
        if pad_len > 0:
            Z_hat = Z_hat[:, :L, :]
            I_cross = I_cross[:, :L, :]
            
        attn_out = Z_hat + (I_cross * self.gate_mem.to(dtype))
        X_res1 = X_res0 + self.post_attention_layernorm(attn_out)
        
        normed_X_ffn = self.pre_feedforward_layernorm(X_res1)
        ffn_out = self.ffn(normed_X_ffn)
        Output = X_res1 + self.post_feedforward_layernorm(ffn_out)
        
        return Output

# ==========================================
# 主模型裝配
# ==========================================

class AGIV2G(nn.Module):
    # 🌟 已切換為 1B 級別優化預設值
    def __init__(self, vocab_size=262144, D=1536, hidden_dim=6144, num_blocks=16, C=256, K=256, M=256, num_heads=6, num_kv_heads=2, head_dim=256):
        super().__init__()
        self.D = D
        self.embedding = nn.Embedding(vocab_size, D)
        self.blocks = nn.ModuleList()
        
        for i in range(num_blocks):
            if (i + 1) % 4 == 0:
                self.blocks.append(AGIV2GlobalBlock(D=D, hidden_dim=hidden_dim, K=K, M=M, C=C, num_heads=num_heads, num_kv_heads=num_kv_heads, head_dim=head_dim))
            else:
                self.blocks.append(AGIV2LocalBlock(D=D, hidden_dim=hidden_dim, C=C, num_heads=num_heads, num_kv_heads=num_kv_heads, head_dim=head_dim))
                
        self.final_norm = GemmaRMSNorm(D)
        self.fc_out = nn.Linear(D, vocab_size, bias=False)
        
        # 🚀 嚴格對齊第一性原理：強制權重綁定 (Weight Tying)
        # 消除 400M 的冗餘參數，確保模型將運算能力集中於 Transformer 核心
        self.fc_out.weight = self.embedding.weight

    def forward(self, x, n_split_index=None):
        out = self.embedding(x)
        out = out * math.sqrt(self.D) 
        for i, block in enumerate(self.blocks):
            shift_size = (block.C // 2) if (i % 2 != 0) else 0
            out = block(out, shift_size=shift_size, n_split_index=n_split_index)
        out = self.final_norm(out)
        logits = self.fc_out(out)
        
        # 🚀 Gemma 3 Logit Soft-Capping
        logits = logits / 30.0
        logits = torch.tanh(logits) * 30.0
        return logits