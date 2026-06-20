import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# 🌟 官方 FlashAttention 導入
try:
    from flash_attn import flash_attn_func
except ImportError:
    raise ImportError(
        "請確定已安裝 flash-attn 套件: pip install flash-attn --no-build-isolation"
    )


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


def apply_rope(x, head_dim, base_freq=1000000.0):
    B, L, num_heads, D = x.shape
    position = torch.arange(L, device=x.device).unsqueeze(1).float()
    div_term = torch.exp(
        torch.arange(0, head_dim, 2, device=x.device).float()
        * -(math.log(base_freq) / head_dim)
    )
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
    hidden_states = hidden_states[:, :, :, None, :].expand(
        B, L, num_kv_heads, n_rep, head_dim
    )
    return hidden_states.reshape(B, L, num_kv_heads * n_rep, head_dim)


# ==========================================
# 核心網路區塊: 原始 GEMMA 3 架構
# ==========================================
class Gemma3DecoderLayer(nn.Module):
    def __init__(
        self,
        D=1152,
        hidden_dim=6912,
        C=1024,
        num_heads=4,
        num_kv_heads=1,
        head_dim=256,
        is_global=False,
        rope_base=1000000.0,
    ):
        super().__init__()
        self.C = C
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_key_value_groups = num_heads // num_kv_heads
        self.is_global = is_global
        self.rope_base = rope_base

        self.input_layernorm = GemmaRMSNorm(D)
        self.post_attention_layernorm = GemmaRMSNorm(D)
        self.pre_feedforward_layernorm = GemmaRMSNorm(D)
        self.post_feedforward_layernorm = GemmaRMSNorm(D)
        self.q_norm = GemmaRMSNorm(head_dim)
        self.k_norm = GemmaRMSNorm(head_dim)

        # 保持原本權重命名以相容 utils.py 的 transplant_and_freeze 移植腳本
        self.W_q_loc = nn.Linear(D, num_heads * self.head_dim, bias=False)
        self.W_k_loc = nn.Linear(D, num_kv_heads * self.head_dim, bias=False)
        self.W_v_loc = nn.Linear(D, num_kv_heads * self.head_dim, bias=False)
        self.o_proj_loc = nn.Linear(num_heads * self.head_dim, D, bias=False)
        self.ffn = GemmaFFN(hidden_size=D, intermediate_size=hidden_dim)

    def forward(self, X, shift_size=0, n_split_index=None):
        B, L, D = X.shape

        normed_X = self.input_layernorm(X)

        Q = self.W_q_loc(normed_X).view(B, L, self.num_heads, self.head_dim)
        K = self.W_k_loc(normed_X).view(B, L, self.num_kv_heads, self.head_dim)
        V = self.W_v_loc(normed_X).view(B, L, self.num_kv_heads, self.head_dim)

        Q = self.q_norm(Q)
        K = self.k_norm(K)

        Q = apply_rope(Q, self.head_dim, base_freq=self.rope_base).contiguous()
        K = apply_rope(K, self.head_dim, base_freq=self.rope_base).contiguous()
        V = V.contiguous()

        # 根據是否為 Global Attention 設定 Window Size
        window_size = (-1, -1) if self.is_global else (self.C - 1, 0)

        attn_out = flash_attn_func(
            Q, K, V, dropout_p=0.0, causal=True, window_size=window_size
        )

        Z_hat = self.o_proj_loc(attn_out.reshape(B, L, self.num_heads * self.head_dim))

        X_res1 = X + self.post_attention_layernorm(Z_hat)
        normed_X_ffn = self.pre_feedforward_layernorm(X_res1)
        ffn_out = self.ffn(normed_X_ffn)
        Output = X_res1 + self.post_feedforward_layernorm(ffn_out)
        return Output


class GEMMA3(nn.Module):
    def __init__(
        self,
        vocab_size=262144,
        D=1152,
        hidden_dim=6912,
        num_blocks=26,
        C=1024,
        num_heads=4,
        num_kv_heads=1,
        head_dim=256,
        rope_local=10000.0,
        rope_global=1000000.0,
    ):
        super().__init__()
        self.D = D
        self.embedding = nn.Embedding(vocab_size, D)
        self.blocks = nn.ModuleList()

        for i in range(num_blocks):
            # 每 6 層設定為 Global Block，其餘為 Local Sliding Window
            is_global = (i + 1) % 6 == 0
            rb = rope_global if is_global else rope_local
            self.blocks.append(
                Gemma3DecoderLayer(
                    D=D,
                    hidden_dim=hidden_dim,
                    C=C,
                    num_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    head_dim=head_dim,
                    is_global=is_global,
                    rope_base=rb,
                )
            )

        self.final_norm = GemmaRMSNorm(D)
        self.fc_out = nn.Linear(D, vocab_size, bias=False)

    def compute_gate_neg_entropy(self):
        if not hasattr(self, "gated_module_list") or not self.gated_module_list:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        
        gated_module = self.gated_module_list[0]
        if not hasattr(gated_module, "_gates_by_layer"):
            return torch.tensor(0.0, device=next(self.parameters()).device)
        
        gates = gated_module._gates_by_layer
        if not gates:
            return torch.tensor(0.0, device=next(self.parameters()).device)
            
        neg_entropy_sum = 0.0
        count = 0
        eps = 1e-8
        
        for layer_idx, gate in list(gates.items()):
            # gate shape: (B, L, 8), values in [0, 1]
            p = torch.stack([gate, 1.0 - gate], dim=0) # shape: (2, B, L, 8)
            p = p / (p.sum(dim=0, keepdim=True) + eps)
            
            entropy = (p * torch.log(p + eps)).sum(dim=0) # shape: (B, L, 8)
            neg_entropy_sum = neg_entropy_sum + entropy.mean()
            count += 1
            
        if count == 0:
            return torch.tensor(0.0, device=next(self.parameters()).device)
            
        gated_module._gates_by_layer.clear()
        return neg_entropy_sum / count


    def forward(self, x, n_split_index=None):
        out = self.embedding(x)
        out = out * math.sqrt(self.D)
        for i, block in enumerate(self.blocks):
            out = block(out, shift_size=0, n_split_index=n_split_index)
        out = self.final_norm(out)
        logits = self.fc_out(out)
        logits = logits / 30.0
        logits = torch.tanh(logits) * 30.0
        return logits


# ============================================================
# DNA Helix Ultimate 16B 雙螺旋架構 (粒子-波動雙流 Gating)
# ============================================================

class DNAHelixDecoderLayer(nn.Module):
    def __init__(
        self,
        D=4096,
        hidden_dim=16384,
        C=1024,
        num_heads=32,
        num_kv_heads=8,
        head_dim=128,
        is_global=False,
        rope_base=1000000.0,
    ):
        super().__init__()
        self.C = C
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.is_global = is_global
        self.rope_base = rope_base
        self.D = D

        self.input_layernorm = GemmaRMSNorm(D)
        self.post_attention_layernorm = GemmaRMSNorm(D)
        self.pre_feedforward_layernorm = GemmaRMSNorm(D)
        self.post_feedforward_layernorm = GemmaRMSNorm(D)
        self.q_norm = GemmaRMSNorm(head_dim)
        self.k_norm = GemmaRMSNorm(head_dim)

        # 粒子頭 (Particle Head - RoPE Window Attention)
        self.W_q_loc = nn.Linear(D, num_heads * head_dim, bias=False)
        self.W_k_loc = nn.Linear(D, num_kv_heads * head_dim, bias=False)
        self.W_v_loc = nn.Linear(D, num_kv_heads * head_dim, bias=False)
        self.o_proj_loc = nn.Linear(num_heads * head_dim, D, bias=False)

        # 波動頭 (Wave Head - Causal Sinc Convolution with Phase Lock)
        self.W_wave = 64
        self.omegas = nn.Parameter(torch.randn(1024))
        self.mlp_H = nn.Sequential(
            nn.Linear(2048, D),
            nn.GELU(),
            nn.Linear(D, D)
        )
        self.fft_norm = GemmaRMSNorm(D)

        # 雙螺旋門控 Router (Gate)
        self.router = nn.Linear(D, 2, bias=False)
        nn.init.normal_(self.router.weight, mean=0.0, std=0.001)
        self.temperature = 2.0

        # FeedForward
        self.ffn = GemmaFFN(hidden_size=D, intermediate_size=hidden_dim)

    def forward(self, X, shift_size=0, n_split_index=None):
        B, L, D = X.shape
        device = X.device
        dtype = X.dtype

        normed_X = self.input_layernorm(X)

        # ─── A. 門控權重與熵 ───
        gate_logits = self.router(normed_X)  # (B, L, 2)
        gate_weights = F.softmax(gate_logits / self.temperature, dim=-1)
        g_particle, g_wave = gate_weights[..., 0:1], gate_weights[..., 1:2]

        eps = 1e-8
        entropy = - (gate_weights * torch.log(gate_weights + eps)).sum(dim=-1).mean()

        # ─── B. 粒子頭 ───
        Q = self.W_q_loc(normed_X).view(B, L, self.num_heads, self.head_dim)
        K = self.W_k_loc(normed_X).view(B, L, self.num_kv_heads, self.head_dim)
        V = self.W_v_loc(normed_X).view(B, L, self.num_kv_heads, self.head_dim)

        Q = self.q_norm(Q)
        K = self.k_norm(K)

        Q = apply_rope(Q, self.head_dim, base_freq=self.rope_base).contiguous()
        K = apply_rope(K, self.head_dim, base_freq=self.rope_base).contiguous()
        V = V.contiguous()

        window_size = (-1, -1) if self.is_global else (self.C - 1, 0)
        attn_out = flash_attn_func(
            Q, K, V, dropout_p=0.0, causal=True, window_size=window_size
        )
        out_particle = self.o_proj_loc(attn_out.reshape(B, L, self.num_heads * self.head_dim))

        # ─── C. 波動頭 ───
        t = torch.arange(self.W_wave, device=device, dtype=torch.float32)
        args = t.unsqueeze(1) * self.omegas.unsqueeze(0).to(torch.float32)
        gamma = torch.cat([torch.sin(args), torch.cos(args)], dim=-1).to(dtype)
        H = self.mlp_H(gamma)

        window = 0.54 - 0.46 * torch.cos(2 * math.pi * t / (self.W_wave - 1))
        window = window.to(dtype).unsqueeze(1)
        H_windowed = H * window

        weight = H_windowed.transpose(0, 1).unsqueeze(1)
        X_wave_norm = self.fft_norm(normed_X)
        X_t = X_wave_norm.transpose(1, 2)
        X_padded = F.pad(X_t, (self.W_wave - 1, 0))
        out_wave = F.conv1d(X_padded, weight, groups=D).transpose(1, 2)

        # ─── D. 雙流混合 ───
        Z_hat = (out_particle * g_particle) + (out_wave * g_wave)

        # ─── E. 殘差與 FFN ───
        X_res1 = X + self.post_attention_layernorm(Z_hat)
        ffn_out = self.ffn(self.pre_feedforward_layernorm(X_res1))
        Output = X_res1 + self.post_feedforward_layernorm(ffn_out)

        return Output, entropy


class DNAHelix(nn.Module):
    def __init__(
        self,
        vocab_size=151936,
        D=4096,
        hidden_dim=16384,
        num_blocks=40,
        C=1024,
        num_heads=32,
        num_kv_heads=8,
        head_dim=128,
        rope_local=10000.0,
        rope_global=1000000.0,
    ):
        super().__init__()
        self.D = D
        self.embedding = nn.Embedding(vocab_size, D)
        self.blocks = nn.ModuleList()

        for i in range(num_blocks):
            is_global = (i + 1) % 6 == 0
            rb = rope_global if is_global else rope_local
            self.blocks.append(
                DNAHelixDecoderLayer(
                    D=D,
                    hidden_dim=hidden_dim,
                    C=C,
                    num_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    head_dim=head_dim,
                    is_global=is_global,
                    rope_base=rb,
                )
            )

        self.final_norm = GemmaRMSNorm(D)
        self.fc_out = nn.Linear(D, vocab_size, bias=False)

        # ─── 完美對齊 Gemma3 Tied Embedding 與權重初始標準化 ───
        self.fc_out.weight = self.embedding.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # Gemma 標準初始化：std = 1 / sqrt(D)
            std = 1.0 / math.sqrt(self.D)
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            std = 1.0 / math.sqrt(self.D)
            nn.init.normal_(module.weight, mean=0.0, std=std)

    def forward(self, x, n_split_index=None):
        out = self.embedding(x)
        out = out * math.sqrt(self.D)
        
        for i, block in enumerate(self.blocks):
            shift_size = (block.C // 2) if (i % 2 != 0) else 0
            out, _ = block(out, shift_size=shift_size, n_split_index=n_split_index)
            
        out = self.final_norm(out)
        logits = self.fc_out(out)
        logits = logits / 30.0
        logits = torch.tanh(logits) * 30.0
        return logits


# ============================================================
# DNA Helix Ultimate 16B 雙螺旋架構 (WPCE 波粒因果糾纏 與 鹼基對子空間路由)
# ============================================================

def apply_phase_locked_rope(x, head_dim, base_freq=1000000.0, delta_theta=None):
    B, L, num_heads, D = x.shape
    position = torch.arange(L, device=x.device).unsqueeze(1).float()
    div_term = torch.exp(
        torch.arange(0, head_dim, 2, device=x.device).float() * -(math.log(base_freq) / head_dim)
    )
    freqs = position * div_term # shape: (L, head_dim // 2)
    emb = freqs.unsqueeze(0).unsqueeze(2) # shape: (1, L, 1, head_dim // 2)
    
    if delta_theta is not None:
        emb = emb + delta_theta # shape: (B, L, 1, head_dim // 2)
        
    emb = torch.cat((emb, emb), dim=-1) # shape: (B, L, 1, head_dim)
    sin_val = torch.sin(emb).to(x.dtype)
    cos_val = torch.cos(emb).to(x.dtype)
    x1 = x[..., : head_dim // 2]
    x2 = x[..., head_dim // 2 :]
    x_rotated = torch.cat((-x2, x1), dim=-1)
    return (x * cos_val) + (x_rotated * sin_val)


def apply_fra_only(x, head_dim, delta_theta=None):
    B, L, num_heads, D = x.shape
    if delta_theta is not None:
        emb = delta_theta
    else:
        return x
    emb = torch.cat((emb, emb), dim=-1) # shape: (B, L, 1, head_dim)
    sin_val = torch.sin(emb).to(x.dtype)
    cos_val = torch.cos(emb).to(x.dtype)
    x1 = x[..., : head_dim // 2]
    x2 = x[..., head_dim // 2 :]
    x_rotated = torch.cat((-x2, x1), dim=-1)
    return (x * cos_val) + (x_rotated * sin_val)


class DNAHelixUltimateGatedModule(nn.Module):
    def __init__(self, D=4096, W=64, slice_size=256, num_heads=32, num_kv_heads=8, head_dim=128):
        super().__init__()
        self.slice_size = slice_size
        self.head_dim = head_dim
        self.W = W
        
        # 1. 波動 Sinc 卷積投射權重
        self.w_layer = nn.Parameter(torch.empty(slice_size, head_dim // 2, dtype=torch.float32))
        # 2. 門控投影權重
        self.w_gate = nn.Parameter(torch.empty(D, 4, dtype=torch.float32))
        
        nn.init.normal_(self.w_layer, mean=0.0, std=0.02)
        nn.init.normal_(self.w_gate, mean=0.0, std=0.02)
        
    def forward(self, x):
        B, L, D = x.shape
        device = x.device
        
        # 1. 獲取波動 Sinc 卷積投射特徵
        x_slice = x[..., :self.slice_size].to(torch.float32)
        w_layer_cast = self.w_layer.to(torch.float32)
        h = torch.matmul(x_slice, w_layer_cast) # (B, L, head_dim // 2)
        
        t = torch.arange(self.W, device=device, dtype=torch.float32)
        t_0 = (self.W - 1) / 2.0
        
        fc = torch.linspace(0.02, 0.48, steps=self.head_dim // 2, device=device, dtype=torch.float32)
        args = 2.0 * fc.unsqueeze(1) * (t.unsqueeze(0) - t_0)
        sinc_val = torch.sinc(args) # [head_dim // 2, W]
        
        window = 0.54 - 0.46 * torch.cos(2.0 * math.pi * t / (self.W - 1))
        window = window.unsqueeze(0)
        
        kernel = 2.0 * fc.unsqueeze(1) * sinc_val * window # [head_dim // 2, W]
        kernel = kernel / (kernel.norm(p=1, dim=1, keepdim=True) + 1e-8)
        
        weight_conv = kernel.unsqueeze(1) # [head_dim // 2, 1, W]
        
        h_t = h.transpose(1, 2)
        h_padded = F.pad(h_t, (self.W - 1, 0))
        
        h_modulated = F.conv1d(h_padded, weight_conv, groups=self.head_dim // 2)
        h_modulated = h_modulated.transpose(1, 2) # [B, L, head_dim // 2]
        
        delta_theta = h_modulated.view(B, L, 1, self.head_dim // 2) # [B, L, 1, head_dim // 2]
        
        # 2. 獲取動態鹼基對閘門投影
        x_cast = x.to(torch.float32)
        w_gate_cast = self.w_gate.to(torch.float32)
        s = torch.matmul(x_cast, w_gate_cast) # (B, L, 4)
        gate = torch.sigmoid(s) # (B, L, 4)
        
        return delta_theta.to(x.dtype), gate.to(x.dtype)


class DNAHelixUltimateDecoderLayer(nn.Module):
    def __init__(
        self,
        D=4096,
        hidden_dim=16384,
        C=1024,
        num_heads=32,
        num_kv_heads=8,
        head_dim=128,
        is_global=False,
        rope_base=1000000.0,
        alpha=0.1,
    ):
        super().__init__()
        self.C = C
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_key_value_groups = num_heads // num_kv_heads
        self.is_global = is_global
        self.rope_base = rope_base
        self.D = D
        self.alpha = alpha

        # 3:1 Head splitting for query heads: 24 Particle, 8 Wave
        self.num_heads_part = (num_heads * 3) // 4
        self.num_heads_wave = num_heads // 4
        
        # 3:1 Head splitting for KV heads
        self.num_kv_heads_part = (num_kv_heads * 3) // 4
        self.num_kv_heads_wave = num_kv_heads // 4

        self.input_layernorm = GemmaRMSNorm(D)
        self.post_attention_layernorm = GemmaRMSNorm(D)
        self.pre_feedforward_layernorm = GemmaRMSNorm(D)
        self.post_feedforward_layernorm = GemmaRMSNorm(D)
        self.q_norm = GemmaRMSNorm(head_dim)
        self.k_norm = GemmaRMSNorm(head_dim)

        self.W_q_loc = nn.Linear(D, num_heads * head_dim, bias=False)
        self.W_k_loc = nn.Linear(D, num_kv_heads * head_dim, bias=False)
        self.W_v_loc = nn.Linear(D, num_kv_heads * head_dim, bias=False)
        self.o_proj_loc = nn.Linear(num_heads * head_dim, D, bias=False)

        # Native DNA Helix Ultimate gated modulator (non-shared per-layer)
        self.gated_module = DNAHelixUltimateGatedModule(
            D=D, W=64, slice_size=D // 16, num_heads=num_heads, num_kv_heads=num_kv_heads, head_dim=head_dim
        )

        self.ffn = GemmaFFN(hidden_size=D, intermediate_size=hidden_dim)

    def forward(self, X, shift_size=0, n_split_index=None):
        B, L, D = X.shape
        device = X.device

        normed_X = self.input_layernorm(X)

        Q = self.W_q_loc(normed_X).view(B, L, self.num_heads, self.head_dim)
        K = self.W_k_loc(normed_X).view(B, L, self.num_kv_heads, self.head_dim)
        V = self.W_v_loc(normed_X).view(B, L, self.num_kv_heads, self.head_dim)

        Q = self.q_norm(Q)
        K = self.k_norm(K)

        K = repeat_kv(K, self.num_key_value_groups) # (B, L, num_heads, head_dim)
        V = repeat_kv(V, self.num_key_value_groups) # (B, L, num_heads, head_dim)

        # 1. 拆分雙鏈：24 個粒子頭 與 8 個波動頭
        Q_part = Q[:, :, :self.num_heads_part]
        Q_wave = Q[:, :, self.num_heads_part:]
        K_part = K[:, :, :self.num_heads_part]
        K_wave = K[:, :, self.num_heads_part:]

        # 2. 幾何粒子鏈：套用標準 RoPE 旋轉
        Q_part = apply_phase_locked_rope(Q_part, self.head_dim, base_freq=self.rope_base, delta_theta=None)
        K_part = apply_phase_locked_rope(K_part, self.head_dim, base_freq=self.rope_base, delta_theta=None)

        # 3. WPCE 因果糾纏引擎
        # Causal 1: 計算粒子性注意力 (AP_Base)，引導波動共振偏置 Δθ
        scores_part_base = torch.matmul(Q_part.transpose(1, 2), K_part.transpose(1, 2).transpose(-1, -2)) / math.sqrt(self.head_dim)
        mask = torch.triu(torch.full((L, L), float('-inf'), device=scores_part_base.device, dtype=scores_part_base.dtype), diagonal=1)
        if not self.is_global:
            row_idx = torch.arange(L, device=device).unsqueeze(1)
            col_idx = torch.arange(L, device=device).unsqueeze(0)
            window_mask = (col_idx < row_idx - self.C + 1)
            mask = mask.masked_fill(window_mask, float('-inf'))

        scores_part = scores_part_base + mask
        AP_Base = torch.softmax(scores_part, dim=-1) # (B, num_heads_part, L, L)

        # 空間特徵幾何聚合
        X_part_aggr = torch.matmul(AP_Base.mean(dim=1), normed_X) # (B, L, D)

        # 動態相位調製 & 門控產生 (粒子性 ➡️ 波動共振)
        delta_theta, gate = self.gated_module(X_part_aggr) # (B, L, 1, head_dim // 2), (B, L, 4)

        # 波動鏈套用動態相位調製
        Q_wave = apply_fra_only(Q_wave, self.head_dim, delta_theta=delta_theta)
        K_wave = apply_fra_only(K_wave, self.head_dim, delta_theta=delta_theta)

        # 計算波動共振相干矩陣 AW
        scores_wave_base = torch.matmul(Q_wave.transpose(1, 2), K_wave.transpose(1, 2).transpose(-1, -2)) / math.sqrt(self.head_dim)
        AW = torch.sigmoid(scores_wave_base + mask) # (B, num_heads_wave, L, L)

        # Causal 2: 波動共振 AW ➡️ 引力聚焦粒子注意力 AP_Entangled
        scores_part_entangled = (scores_part_base * (1.0 + self.alpha * AW.mean(dim=1, keepdim=True))) + mask
        AP_Entangled = torch.softmax(scores_part_entangled, dim=-1) # (B, num_heads_part, L, L)

        # 4. 特徵融合與 V 相乘
        V_part = V[:, :, :self.num_heads_part].transpose(1, 2)
        attn_part = torch.matmul(AP_Entangled, V_part).transpose(1, 2) # (B, L, num_heads_part, head_dim)

        V_wave = V[:, :, self.num_heads_part:].transpose(1, 2)
        attn_wave = torch.matmul(AW * AP_Base.mean(dim=1, keepdim=True), V_wave).transpose(1, 2) # (B, L, num_heads_wave, head_dim)

        # 5. 同層雙螺旋正交子空間路由 (Ultimate 4-subspace 版本)
        subspace_dim = self.head_dim // 4
        attn_part_grouped = attn_part.view(B, L, self.num_heads_part, 4, subspace_dim)
        attn_wave_grouped = attn_wave.view(B, L, self.num_heads_wave, 4, subspace_dim)

        gate_w = gate.unsqueeze(2).unsqueeze(-1) # shape: (B, L, 1, 4, 1)
        gate_p = 1.0 - gate_w

        attn_part_gated = (attn_part_grouped * gate_p).reshape(B, L, self.num_heads_part, self.head_dim)
        attn_wave_gated = (attn_wave_grouped * gate_w).reshape(B, L, self.num_heads_wave, self.head_dim)

        attn_out_gated = torch.cat([attn_part_gated, attn_wave_gated], dim=2) # (B, L, num_heads, head_dim)

        # 6. 重組投影
        Z_hat = self.o_proj_loc(attn_out_gated.reshape(B, L, self.num_heads * self.head_dim))

        X_res1 = X + self.post_attention_layernorm(Z_hat)
        normed_X_ffn = self.pre_feedforward_layernorm(X_res1)
        ffn_out = self.ffn(normed_X_ffn)
        Output = X_res1 + self.post_feedforward_layernorm(ffn_out)

        # 7. 計算閘門負熵
        with torch.autocast(device_type="cuda", enabled=False):
            eps = 1e-8
            gate_32 = gate.to(torch.float32)
            p = torch.stack([gate_32, 1.0 - gate_32], dim=0)
            p = p / (p.sum(dim=0, keepdim=True) + eps)
            block_entropy = (p * torch.log(p + eps)).sum(dim=0).mean()

        return Output, block_entropy.to(X.dtype)


class DNAHelixUltimate(nn.Module):
    def __init__(
        self,
        vocab_size=151936,
        D=4096,
        hidden_dim=16384,
        num_blocks=40,
        C=1024,
        num_heads=32,
        num_kv_heads=8,
        head_dim=128,
        rope_local=10000.0,
        rope_global=1000000.0,
        alpha=0.1,
    ):
        super().__init__()
        self.D = D
        self.embedding = nn.Embedding(vocab_size, D)
        self.blocks = nn.ModuleList()

        for i in range(num_blocks):
            is_global = (i + 1) % 6 == 0
            rb = rope_global if is_global else rope_local
            self.blocks.append(
                DNAHelixUltimateDecoderLayer(
                    D=D,
                    hidden_dim=hidden_dim,
                    C=C,
                    num_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    head_dim=head_dim,
                    is_global=is_global,
                    rope_base=rb,
                    alpha=alpha,
                )
            )

        self.final_norm = GemmaRMSNorm(D)
        self.fc_out = nn.Linear(D, vocab_size, bias=False)

        # Tied Embedding & Initialization
        self.fc_out.weight = self.embedding.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 1.0 / math.sqrt(self.D)
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            std = 1.0 / math.sqrt(self.D)
            nn.init.normal_(module.weight, mean=0.0, std=std)

    def forward(self, x, n_split_index=None):
        out = self.embedding(x)
        out = out * math.sqrt(self.D)
        
        for i, block in enumerate(self.blocks):
            shift_size = (block.C // 2) if (i % 2 != 0) else 0
            out, _ = block(out, shift_size=shift_size, n_split_index=n_split_index)
            
        out = self.final_norm(out)
        logits = self.fc_out(out)
        logits = logits / 30.0
        logits = torch.tanh(logits) * 30.0
        return logits


