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
# 核心網路區塊
# ==========================================


class PureFFTBlock(nn.Module):
    def __init__(self, D=1152, K=1024, W=64):
        super().__init__()
        self.fft_norm = GemmaRMSNorm(D)
        self.W = W
        self.omegas = nn.Parameter(torch.randn(K))
        self.mlp_H = nn.Sequential(nn.Linear(2 * K, D), nn.GELU(), nn.Linear(D, D))

    def forward(self, X, shift_size=0, n_split_index=None):
        B, L, D = X.shape
        device = X.device
        dtype = X.dtype

        # Apply RMSNorm
        normed_X = self.fft_norm(X)

        # Generate the expectation wave H of length W (sliding window size)
        t = torch.arange(self.W, device=device, dtype=torch.float32)
        args = t.unsqueeze(1) * self.omegas.unsqueeze(0).to(torch.float32)
        gamma = torch.cat([torch.sin(args), torch.cos(args)], dim=-1).to(dtype)
        H = self.mlp_H(gamma)  # [W, D]

        # Apply causal windowing to H (Hamming window)
        t_w = torch.arange(self.W, device=device, dtype=torch.float32)
        window = 0.54 - 0.46 * torch.cos(2 * math.pi * t_w / (self.W - 1))
        window = window.to(dtype).unsqueeze(1)  # [W, 1]
        H_windowed = H * window  # [W, D]

        # Depthwise conv1d (No flip needed because F.conv1d is cross-correlation)
        weight = H_windowed.transpose(0, 1).unsqueeze(1)  # [D, 1, W]
        X_t = normed_X.transpose(1, 2)  # [B, D, L]
        X_padded = F.pad(X_t, (self.W - 1, 0))  # [B, D, L + W - 1]
        
        Y_t = F.conv1d(X_padded, weight, groups=D)  # [B, D, L]
        return X + Y_t.transpose(1, 2)


class DynamicPhaseLockFFTBlock(nn.Module):
    """
    動態相位鎖定 Causal Sinc Block（資料驅動版，參數量與 PureFFTBlock 100% 相同）

    使用時域 Causal Sinc 卷積代替頻域 RFFT，完全杜絕未來資訊洩漏。
    利用自迴歸因果互相關動態提取時域位移量（相位鎖定），無須任何額外參數。
    """
    def __init__(self, D=1152, K=1024, W=64):
        super().__init__()
        self.fft_norm = GemmaRMSNorm(D)
        self.W = W
        self.omegas = nn.Parameter(torch.randn(K))
        self.mlp_H = nn.Sequential(nn.Linear(2 * K, D), nn.GELU(), nn.Linear(D, D))
        self.eps = 1e-8

    def forward(self, X, shift_size=0, n_split_index=None):
        B, L, D = X.shape
        device = X.device
        dtype = X.dtype

        # Apply RMSNorm
        normed_X = self.fft_norm(X)

        # 1. Generate the expectation wave H of length W (sliding window size)
        t = torch.arange(self.W, device=device, dtype=torch.float32)
        args = t.unsqueeze(1) * self.omegas.unsqueeze(0).to(torch.float32)
        gamma = torch.cat([torch.sin(args), torch.cos(args)], dim=-1).to(dtype)
        H = self.mlp_H(gamma)  # [W, D]

        # 2. Causal Left-padding and unfolding
        X_t = normed_X.transpose(1, 2)  # [B, D, L]
        X_padded = F.pad(X_t, (self.W - 1, 0))  # [B, D, L + W - 1]
        X_unfold = X_padded.unfold(2, self.W, 1)  # [B, D, L, W]

        # 3. Dynamic Delay Extraction via causal cross-correlation with H (using fast Conv1d)
        H_exp = H.transpose(0, 1).unsqueeze(1)  # [D, 1, W]
        corr_t = F.conv1d(X_padded, H_exp, groups=D)  # [B, D, L]
        
        # Scale delay to a reasonable range (e.g., max shift of W // 4)
        max_delay = float(self.W // 4)
        tau_d_t = torch.tanh(corr_t) * max_delay  # [B, D, L]

        # 4. Generate Causal Sinc window weights dynamically in [B, D, L, W] layout
        grid = torch.arange(self.W, device=device, dtype=torch.float32).view(1, 1, 1, self.W)
        t_w = torch.arange(self.W, device=device, dtype=torch.float32)
        window = 0.54 - 0.46 * torch.cos(2 * math.pi * t_w / (self.W - 1))
        window = window.to(dtype).view(1, 1, 1, self.W)

        # Compute sinc(grid - tau_d_t)
        delta = grid - tau_d_t.unsqueeze(-1)  # [B, D, L, W]
        sinc_weights = torch.sinc(delta).to(dtype)
        H_filter = sinc_weights * window  # [B, D, L, W]

        # 5. Apply dynamic filter to the unfolded input
        Y_t = torch.sum(X_unfold * H_filter, dim=-1)  # [B, D, L]
        return X + Y_t.transpose(1, 2)



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


class AGIGEMMA3(nn.Module):
    def __init__(
        self,
        vocab_size=262144,
        D=1152,
        hidden_dim=6912,
        num_blocks=26,
        N_fft_F=1,
        N_fft_B=1,
        C=1024,
        K=1024,
        num_heads=4,
        num_kv_heads=1,
        head_dim=256,
        rope_local=10000.0,
        rope_global=1000000.0,
        fft_block_type="pure",  # "pure" = 原始 PureFFTBlock, "dynamic" = 動態相位鎖定
    ):
        super().__init__()
        self.D = D
        self.embedding = nn.Embedding(vocab_size, D)

        # 選擇 FFT Block 類型
        if fft_block_type == "dynamic":
            _make_fft = lambda: DynamicPhaseLockFFTBlock(D=D, K=K)
        else:
            _make_fft = lambda: PureFFTBlock(D=D, K=K)

        # 前 N 層 FFT
        self.pre_fft_blocks = nn.ModuleList(
            [_make_fft() for _ in range(N_fft_F)]
        )

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

        # 後 N 層 FFT
        self.post_fft_blocks = nn.ModuleList(
            [_make_fft() for _ in range(N_fft_B)]
        )

        self.final_norm = GemmaRMSNorm(D)
        self.fc_out = nn.Linear(D, vocab_size, bias=False)

    def forward(self, x, n_split_index=None):
        out = self.embedding(x)
        out = out * math.sqrt(self.D)

        for block in self.pre_fft_blocks:
            out = block(out, shift_size=0, n_split_index=n_split_index)

        for block in self.blocks:
            out = block(out, shift_size=0, n_split_index=n_split_index)

        for block in self.post_fft_blocks:
            out = block(out, shift_size=0, n_split_index=n_split_index)

        out = self.final_norm(out)
        logits = self.fc_out(out)
        logits = logits / 30.0
        logits = torch.tanh(logits) * 30.0
        return logits
