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
# 基礎 SOTA 元件庫 (對齊 Gemma 3 物理特性)
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
    B, L, num_heads, D = x.shape
    position = torch.arange(L, device=x.device).unsqueeze(1).float()
    div_term = torch.exp(
        torch.arange(0, head_dim, 2, device=x.device).float()
        * -(math.log(1000000.0) / head_dim)
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
# 核心網路區塊 (AGIV2G 核心邏輯繼承)
# ==========================================
class AGIV2LocalBlock(nn.Module):
    def __init__(
        self, D=1152, hidden_dim=6912, C=256, num_heads=4, num_kv_heads=1, head_dim=256
    ):
        super().__init__()
        self.C = C
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
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

    def forward(self, X, **kwargs):
        B, L, D = X.shape
        normed_X = self.input_layernorm(X)
        Q = self.W_q_loc(normed_X).view(B, L, self.num_heads, self.head_dim)
        K = self.W_k_loc(normed_X).view(B, L, self.num_kv_heads, self.head_dim)
        V = self.W_v_loc(normed_X).view(B, L, self.num_kv_heads, self.head_dim)
        Q, K = self.q_norm(Q), self.k_norm(K)
        Q = apply_rope(Q, self.head_dim).contiguous()
        K = apply_rope(K, self.head_dim).contiguous()
        V = V.contiguous()
        attn_out = flash_attn_func(
            Q, K, V, dropout_p=0.0, causal=True, window_size=(self.C - 1, 0)
        )
        Z_hat = self.o_proj_loc(attn_out.reshape(B, L, self.num_heads * self.head_dim))
        X_res1 = X + self.post_attention_layernorm(Z_hat)
        return X_res1 + self.post_feedforward_layernorm(
            self.ffn(self.pre_feedforward_layernorm(X_res1))
        )


class AGIV2GlobalBlock(nn.Module):
    def __init__(
        self,
        D=1152,
        hidden_dim=6912,
        K=1024,
        M=1024,
        C=256,
        num_heads=4,
        num_kv_heads=1,
        head_dim=256,
    ):
        super().__init__()
        self.D, self.C, self.M = D, C, M
        self.num_heads, self.num_kv_heads, self.head_dim = (
            num_heads,
            num_kv_heads,
            head_dim,
        )
        self.num_key_value_groups = num_heads // num_kv_heads
        self.input_layernorm = GemmaRMSNorm(D)
        self.post_attention_layernorm = GemmaRMSNorm(D)
        self.pre_feedforward_layernorm = GemmaRMSNorm(D)
        self.post_feedforward_layernorm = GemmaRMSNorm(D)
        self.fft_norm, self.mem_norm = GemmaRMSNorm(D), GemmaRMSNorm(D)
        self.q_norm, self.k_norm = GemmaRMSNorm(head_dim), GemmaRMSNorm(head_dim)
        self.router = nn.Linear(D, 3, bias=False)
        nn.init.normal_(self.router.weight, mean=0.0, std=0.001)
        self.register_buffer("temperature", torch.tensor(2.0))
        self.omegas = nn.Parameter(torch.randn(K))
        self.mlp_H = nn.Sequential(nn.Linear(2 * K, D), nn.GELU(), nn.Linear(D, D))
        self.Q_mem = nn.Parameter(torch.randn(M, D))
        self.W_k_mem, self.W_v_mem = nn.Linear(D, D, bias=False), nn.Linear(
            D, D, bias=False
        )
        self.W_q_loc = nn.Linear(D, num_heads * self.head_dim, bias=False)
        self.W_k_loc = nn.Linear(D, num_kv_heads * self.head_dim, bias=False)
        self.W_v_loc = nn.Linear(D, num_kv_heads * self.head_dim, bias=False)
        self.o_proj_loc = nn.Linear(num_heads * self.head_dim, D, bias=False)
        self.W_q_cross = nn.Linear(D, num_heads * self.head_dim, bias=False)
        self.W_k_cross = nn.Linear(D, num_kv_heads * self.head_dim, bias=False)
        self.W_v_cross = nn.Linear(D, num_kv_heads * self.head_dim, bias=False)
        self.o_proj_cross = nn.Linear(num_heads * self.head_dim, D, bias=False)
        self.ffn = GemmaFFN(hidden_size=D, intermediate_size=hidden_dim)

    def forward(self, X, n_split_index=None, **kwargs):
        B, L, D = X.shape
        dtype, device = X.dtype, X.device
        gate_logits = self.router(X)
        routing_weights = torch.sigmoid(gate_logits / self.temperature)
        g_loc, g_mem, g_fft = (
            routing_weights[..., 0:1],
            routing_weights[..., 1:2],
            routing_weights[..., 2:3],
        )
        scale = (g_loc + g_mem + g_fft).clamp(min=1.0)
        g_loc, g_mem, g_fft = g_loc / scale, g_mem / scale, g_fft / scale
        self.avg_g_loc, self.avg_g_mem, self.avg_g_fft = (
            g_loc.mean().detach(),
            g_mem.mean().detach(),
            g_fft.mean().detach(),
        )
        # 保留帶梯度版本供負熵損失使用
        self._g_loc_for_loss = g_loc.mean()
        self._g_mem_for_loss = g_mem.mean()
        self._g_fft_for_loss = g_fft.mean()

        # Phase I: FFT
        if n_split_index is not None:
            mask = (
                (
                    torch.arange(L, device=device).unsqueeze(0)
                    < n_split_index.unsqueeze(1)
                )
                .unsqueeze(-1)
                .to(dtype)
            )
            X_past = X * mask
        else:
            X_past, mask = X, None

        t = torch.arange(L, device=device, dtype=torch.float32)
        args = t.unsqueeze(1) * self.omegas.unsqueeze(0).to(torch.float32)
        gamma = torch.cat([torch.sin(args), torch.cos(args)], dim=-1).to(dtype)
        H = self.mlp_H(gamma)
        X_f = torch.fft.rfft(
            F.pad(self.fft_norm(X_past), (0, 0, 0, L)).to(torch.float32), dim=1
        )
        H_f = torch.fft.rfft(F.pad(H, (0, 0, 0, L)).to(torch.float32), dim=0).unsqueeze(
            0
        )
        Y_sys1 = torch.fft.irfft(X_f * torch.conj(H_f), n=2 * L, dim=1).to(dtype)[
            :, :L, :
        ]
        X_res0 = X + (Y_sys1 * g_fft)

        # Phase II: Memory
        normed_X_mem = self.mem_norm(X_res0 * mask if mask is not None else X_res0)
        K_m, V_m = self.W_k_mem(normed_X_mem), self.W_v_mem(normed_X_mem)
        pad = (self.C - L % self.C) % self.C
        K_mp, V_mp = F.pad(K_m, (0, 0, 0, pad)), F.pad(V_m, (0, 0, 0, pad))
        N_m = K_mp.size(1) // self.C
        Q_exp = self.Q_mem.view(1, 1, self.M, D).expand(B, N_m, -1, -1)
        upd = F.scaled_dot_product_attention(
            Q_exp.reshape(B * N_m, 1, self.M, D),
            K_mp.view(B * N_m, 1, self.C, D),
            V_mp.view(B * N_m, 1, self.C, D),
            is_causal=False,
        )
        M_glob = (
            upd.reshape(B, N_m, self.M, D)
            .mean(dim=1, keepdim=True)
            .expand(-1, N_m, -1, -1)
        )

        # Phase III: Local + Cross
        norm_X_loc = self.input_layernorm(X_res0)
        QL = self.q_norm(
            self.W_q_loc(norm_X_loc).view(B, L, self.num_heads, self.head_dim)
        )
        KL = self.k_norm(
            self.W_k_loc(norm_X_loc).view(B, L, self.num_kv_heads, self.head_dim)
        )
        VL = self.W_v_loc(norm_X_loc).view(B, L, self.num_kv_heads, self.head_dim)
        attn_loc = flash_attn_func(
            apply_rope(QL, self.head_dim).contiguous(),
            apply_rope(KL, self.head_dim).contiguous(),
            VL.contiguous(),
            causal=True,
            window_size=(self.C - 1, 0),
        )
        ZH = self.o_proj_loc(attn_loc.reshape(B, L, -1))
        ZH_p = F.pad(ZH, (0, 0, 0, pad)).view(B, N_m, self.C, -1)
        QC = (
            self.W_q_cross(ZH_p)
            .view(B * N_m, self.C, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        KC = repeat_kv(
            self.W_k_cross(M_glob).view(
                B * N_m, self.M, self.num_kv_heads, self.head_dim
            ),
            self.num_key_value_groups,
        ).transpose(1, 2)
        VC = repeat_kv(
            self.W_v_cross(M_glob).view(
                B * N_m, self.M, self.num_kv_heads, self.head_dim
            ),
            self.num_key_value_groups,
        ).transpose(1, 2)
        IC = (
            F.scaled_dot_product_attention(QC, KC, VC, is_causal=False)
            .transpose(1, 2)
            .reshape(B, -1, self.num_heads * self.head_dim)
        )
        X_res1 = X_res0 + self.post_attention_layernorm(
            (ZH * g_loc) + (self.o_proj_cross(IC[:, :L, :]) * g_mem)
        )
        return X_res1 + self.post_feedforward_layernorm(
            self.ffn(self.pre_feedforward_layernorm(X_res1))
        )


# ==========================================
# 終極融合：AGIV3 (純 Byte 態漏斗架構)
# ==========================================
class AGIV3(nn.Module):
    def __init__(
        self,
        vocab_size=256,
        D_shallow=768,
        D_main=1152,
        hidden_shallow=4608,
        hidden_main=6912,
        num_enc=4,
        num_main=26,
        num_dec=4,
        C=256,
        K=1024,
        M=2048,
        head_dim=256,
    ):
        super().__init__()
        self.D_main = D_main
        self.embedding = nn.Embedding(vocab_size, D_shallow)
        self.encoder_blocks = nn.ModuleList(
            [
                AGIV2LocalBlock(
                    D=D_shallow,
                    hidden_dim=hidden_shallow,
                    C=C,
                    num_heads=3,
                    head_dim=head_dim,
                )
                for _ in range(num_enc)
            ]
        )
        # 🔒 Causal collapse：padding=0，forward 時手動左側補 (kernel_size - stride) = 4
        self.collapse = nn.Conv1d(D_shallow, D_main, kernel_size=8, stride=4, padding=0)
        self.main_blocks = nn.ModuleList()
        for i in range(num_main):
            if (i + 1) % 6 == 0:
                self.main_blocks.append(
                    AGIV2GlobalBlock(D=D_main, hidden_dim=hidden_main, K=K, M=M, C=C)
                )
            else:
                self.main_blocks.append(
                    AGIV2LocalBlock(D=D_main, hidden_dim=hidden_main, C=C)
                )
        self.expand = nn.ConvTranspose1d(D_main, D_shallow, kernel_size=4, stride=4)
        self.decoder_blocks = nn.ModuleList(
            [
                AGIV2LocalBlock(
                    D=D_shallow,
                    hidden_dim=hidden_shallow,
                    C=C,
                    num_heads=3,
                    head_dim=head_dim,
                )
                for _ in range(num_dec)
            ]
        )
        self.final_norm = GemmaRMSNorm(D_shallow)
        self.fc_out = nn.Linear(D_shallow, vocab_size, bias=False)

    def set_temperature(self, temp):
        for b in self.main_blocks:
            if hasattr(b, "temperature"):
                b.temperature.fill_(temp)

    def compute_gate_neg_entropy(self):
        """計算所有 GlobalBlock 閘門的負熵損失，防止任何閘門完全歸零。
        返回帶梯度的標量，可直接加入總 loss 反向傳播。
        """
        neg_entropy_sum = 0.0
        count = 0
        eps = 1e-8
        for block in self.main_blocks:
            if hasattr(block, "_g_loc_for_loss"):
                p = torch.stack(
                    [
                        block._g_loc_for_loss,
                        block._g_mem_for_loss,
                        block._g_fft_for_loss,
                    ]
                )
                p = p / (p.sum() + eps)
                neg_entropy_sum = neg_entropy_sum + (p * torch.log(p + eps)).sum()
                count += 1
        if count == 0:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        return neg_entropy_sum / count

    def forward(self, x, n_split_index=None):
        B, L_byte = x.shape
        pad_len = (4 - (L_byte % 4)) % 4
        if pad_len > 0:
            x = F.pad(x, (0, pad_len), value=0)
        L_padded = x.shape[1]

        out = self.embedding(x) * math.sqrt(768)
        for b in self.encoder_blocks:
            out = b(out)

        # 🔒 Skip connection：保存 encoder 因果輸出，expand 後補回細節
        enc_out = out  # [B, L_padded, D_shallow]，position i 只含 bytes [0, i]

        # 🔒 Causal collapse：左側補 (kernel_size - stride) = 4，右側不補
        causal_pad = 4  # kernel_size(8) - stride(4)
        out_t = F.pad(out.transpose(1, 2), (causal_pad, 0))
        out = self.collapse(out_t).transpose(1, 2)  # [B, L//4, D_main]

        # 🌟 主線處理，支援 n_split_index
        # Causal collapse 後：latent j 含 original bytes [4j-4, 4j+3] 的資訊
        if n_split_index is not None:
            adj_n_split = torch.clamp((n_split_index - 3) // 4, min=0)
        else:
            adj_n_split = None
        for b in self.main_blocks:
            out = b(out, n_split_index=adj_n_split)

        out = self.expand(out.transpose(1, 2)).transpose(
            1, 2
        )  # [B, >=L_padded, D_shallow]
        out = out[:, :L_padded, :]

        # 🔒 Causal expand delay：右移 (stride-1)=3 位，防止組內未來洩漏
        # 原始 expand: position i 含 latent i//4 的資訊 = bytes [0, 4(i//4)+3]
        # byte 4j 會看到 bytes 4j+1..4j+3（組內未來）→ 必須移除
        # 右移 3 後: position i 含 latent (i-3)//4 = bytes [0, 4((i-3)//4)+3]
        # 驗證: pos 3→latent 0→bytes[0,3]→預測byte4 ✓, pos 7→latent 1→bytes[0,7]→預測byte8 ✓
        causal_delay = 3  # stride(4) - 1
        out = F.pad(out, (0, 0, causal_delay, 0))[:, :L_padded, :]

        # 🔒 Skip connection：encoder 的因果細節 + expand 的全域潛空間上下文
        out = out + enc_out

        for b in self.decoder_blocks:
            out = b(out)
        logits = self.fc_out(self.final_norm(out))
        logits = torch.tanh(logits / 30.0) * 30.0
        return logits[:, :-pad_len, :] if pad_len > 0 else logits
