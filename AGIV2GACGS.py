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
class AGIV2LocalBlock(nn.Module):
    def __init__(
        self,
        D=1152,
        hidden_dim=6912,
        C=1024,
        num_heads=4,
        num_kv_heads=1,
        head_dim=256,
        rope_base=1000000.0,
    ):
        super().__init__()
        self.C = C
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_key_value_groups = num_heads // num_kv_heads
        self.rope_base = rope_base

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

        normed_X = self.input_layernorm(X)

        # 🌟 直接提取 Q, K, V，不手動 Padding
        Q = self.W_q_loc(normed_X).view(B, L, self.num_heads, self.head_dim)
        K = self.W_k_loc(normed_X).view(B, L, self.num_kv_heads, self.head_dim)
        V = self.W_v_loc(normed_X).view(B, L, self.num_kv_heads, self.head_dim)

        Q = self.q_norm(Q)
        K = self.k_norm(K)

        # 確保記憶體連續性 (Contiguous) 以符合 FlashAttention 底層要求
        Q = apply_rope(Q, self.head_dim, base_freq=self.rope_base).contiguous()
        K = apply_rope(K, self.head_dim, base_freq=self.rope_base).contiguous()
        V = V.contiguous()

        # 🌟 OOM 免疫與極致加速：調用官方 FlashAttention，使用原生因果滑動窗口
        attn_out = flash_attn_func(
            Q, K, V, dropout_p=0.0, causal=True, window_size=(self.C - 1, 0)
        )

        # 🌟 核心修正：將 1024 維的工作空間，映射回 D = 1152 的主線空間
        Z_hat = self.o_proj_loc(attn_out.reshape(B, L, self.num_heads * self.head_dim))

        X_res1 = X + self.post_attention_layernorm(Z_hat)
        normed_X_ffn = self.pre_feedforward_layernorm(X_res1)
        ffn_out = self.ffn(normed_X_ffn)
        Output = X_res1 + self.post_feedforward_layernorm(ffn_out)
        return Output


class AGIV2GlobalBlock(nn.Module):
    def __init__(
        self,
        D=1152,
        hidden_dim=6912,
        K=1024,
        M=1024,
        C=1024,
        num_heads=4,
        num_kv_heads=1,
        head_dim=256,
        rope_base=1000000.0,
    ):
        super().__init__()
        self.D = D
        self.C = C
        self.M = M
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_key_value_groups = num_heads // num_kv_heads
        self.rope_base = rope_base

        self.input_layernorm = GemmaRMSNorm(D)
        self.post_attention_layernorm = GemmaRMSNorm(D)
        self.pre_feedforward_layernorm = GemmaRMSNorm(D)
        self.post_feedforward_layernorm = GemmaRMSNorm(D)
        self.mem_norm = GemmaRMSNorm(D)
        self.q_norm = GemmaRMSNorm(head_dim)
        self.k_norm = GemmaRMSNorm(head_dim)

        self.router = nn.Linear(D, 2, bias=False)
        nn.init.normal_(self.router.weight, mean=0.0, std=0.001)
        self.register_buffer("temperature", torch.tensor(2.0))

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

        # 納許均衡路由分配
        gate_logits = self.router(X)
        routing_weights = F.sigmoid(gate_logits / self.temperature)
        g_loc = routing_weights[..., 0:1].to(dtype)
        g_mem = routing_weights[..., 1:2].to(dtype)

        # 動態能量守恆校準 (防止特徵爆炸)
        gate_sum = g_loc + g_mem
        scale_factor = gate_sum.clamp(min=1.0)

        g_loc = (g_loc / scale_factor).to(dtype)
        g_mem = (g_mem / scale_factor).to(dtype)

        self.avg_g_loc = g_loc.mean().detach()
        self.avg_g_mem = g_mem.mean().detach()
        # 保留帶梯度版本供負熵損失使用
        self._g_loc_for_loss = g_loc.mean()
        self._g_mem_for_loss = g_mem.mean()

        if n_split_index is not None:
            seq_range = torch.arange(L, device=device).unsqueeze(0)
            past_mask = seq_range < n_split_index.unsqueeze(1)
            X_past = X * past_mask.unsqueeze(-1).to(dtype)
        else:
            X_past = X
            past_mask = None

        # ==========================================
        # [Phase II] 潛在記憶池 (Phase I FFT 已移除)
        # ==========================================
        X_res0 = X
        X_res0_past = X_past

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
                dropout_p=0.0,
                is_causal=False,
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
                dropout_p=0.0,
                is_causal=False,
            ).reshape(B, N_mem, self.M, D)
            M_global_single = updates.mean(dim=1, keepdim=True)

        M_global = M_global_single.expand(-1, N_mem, -1, -1)

        # ==========================================
        # [Phase III] Local + Cross 注意力
        # ==========================================
        normed_X_loc = self.input_layernorm(X_res0)

        Q_loc = self.W_q_loc(normed_X_loc).view(B, L, self.num_heads, self.head_dim)
        K_loc = self.W_k_loc(normed_X_loc).view(B, L, self.num_kv_heads, self.head_dim)
        V_loc = self.W_v_loc(normed_X_loc).view(B, L, self.num_kv_heads, self.head_dim)

        Q_loc = self.q_norm(Q_loc)
        K_loc = self.k_norm(K_loc)

        Q_loc = apply_rope(Q_loc, self.head_dim, base_freq=self.rope_base).contiguous()
        K_loc = apply_rope(K_loc, self.head_dim, base_freq=self.rope_base).contiguous()
        V_loc = V_loc.contiguous()

        # 🌟 使用 FlashAttention 處理 L x L 的局部視角
        attn_out_loc = flash_attn_func(
            Q_loc, K_loc, V_loc, dropout_p=0.0, causal=True, window_size=(self.C - 1, 0)
        )

        # 🌟 核心修正：正確映射回 D = 1152
        Z_hat = self.o_proj_loc(
            attn_out_loc.reshape(B, L, self.num_heads * self.head_dim)
        )

        # --- 為了與 N_mem 對齊進行 Cross Attention，對 Z_hat 進行補齊 ---
        Z_hat_pad = F.pad(Z_hat, (0, 0, 0, mem_pad)) if mem_pad > 0 else Z_hat
        Z_hat_chunks = Z_hat_pad.view(B, N_mem, self.C, -1)

        Q_cross = self.W_q_cross(Z_hat_chunks).view(
            B * N_mem, self.C, self.num_heads, self.head_dim
        )
        K_cross = self.W_k_cross(M_global).view(
            B * N_mem, self.M, self.num_kv_heads, self.head_dim
        )
        V_cross = self.W_v_cross(M_global).view(
            B * N_mem, self.M, self.num_kv_heads, self.head_dim
        )

        K_cross = repeat_kv(K_cross, self.num_key_value_groups)
        V_cross = repeat_kv(V_cross, self.num_key_value_groups)

        Q_cross = Q_cross.transpose(1, 2)
        K_cross = K_cross.transpose(1, 2)
        V_cross = V_cross.transpose(1, 2)

        I_cross = F.scaled_dot_product_attention(
            Q_cross, K_cross, V_cross, dropout_p=0.0, is_causal=False
        )
        I_cross = I_cross.transpose(1, 2).reshape(
            B, L_mem_pad, self.num_heads * self.head_dim
        )
        I_cross = self.o_proj_cross(I_cross)

        if mem_pad > 0:
            I_cross = I_cross[:, :L, :]

        attn_out = (Z_hat * g_loc) + (I_cross * g_mem)
        X_res1 = X_res0 + self.post_attention_layernorm(attn_out)

        normed_X_ffn = self.pre_feedforward_layernorm(X_res1)
        ffn_out = self.ffn(normed_X_ffn)
        Output = X_res1 + self.post_feedforward_layernorm(ffn_out)
        return Output


class AGIV2G(nn.Module):
    def __init__(
        self,
        vocab_size=262144,
        D=1152,
        hidden_dim=6912,
        num_blocks=26,
        C=1024,
        K=1024,
        M=1024,
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
            if (i + 1) % 6 == 0:
                self.blocks.append(
                    AGIV2GlobalBlock(
                        D=D,
                        hidden_dim=hidden_dim,
                        K=K,
                        M=M,
                        C=C,
                        num_heads=num_heads,
                        num_kv_heads=num_kv_heads,
                        head_dim=head_dim,
                        rope_base=rope_global,
                    )
                )
            else:
                self.blocks.append(
                    AGIV2LocalBlock(
                        D=D,
                        hidden_dim=hidden_dim,
                        C=C,
                        num_heads=num_heads,
                        num_kv_heads=num_kv_heads,
                        head_dim=head_dim,
                        rope_base=rope_local,
                    )
                )

        self.final_norm = GemmaRMSNorm(D)
        self.fc_out = nn.Linear(D, vocab_size, bias=False)

    def set_temperature(self, temp):
        for block in self.blocks:
            if hasattr(block, "temperature"):
                block.temperature.fill_(temp)

    def compute_gate_neg_entropy(self):
        """計算所有 GlobalBlock 閘門的負熵損失，防止任何閘門完全歸零。

        負熵越大 (接近 0) 代表分布越集中 → 懲罰越大
        負熵越小 代表分布越均勻 → 懲罰越小
        返回帶梯度的標量，可直接加入總 loss 反向傳播。
        """
        neg_entropy_sum = 0.0
        count = 0
        eps = 1e-8
        for block in self.blocks:
            if hasattr(block, "_g_loc_for_loss"):
                # 使用帶梯度的閘門均值
                p = torch.stack([block._g_loc_for_loss, block._g_mem_for_loss])
                # 歸一化為機率分布
                p = p / (p.sum() + eps)
                # 負熵: sum(p * log(p))
                neg_entropy_sum = neg_entropy_sum + (p * torch.log(p + eps)).sum()
                count += 1
        if count == 0:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        # 取平均，返回負熵 (越接近 0 懲罰越大)
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
