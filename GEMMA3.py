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


