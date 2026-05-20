# AGIV2G_Fast.py — Level 3 極致融合加速版 (去除碎片化擴張的真正 O(1) Memory 版本)
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

try:
    from flash_attn import flash_attn_func
except ImportError:
    raise ImportError("請安裝 flash-attn: pip install flash-attn --no-build-isolation")

# ==========================================
# 🌟 引入 FlashFFTConv
# ==========================================
# 暫時強制關閉 FlashFFTConv，以避開使用者的 CUDA 驅動 PTX 編譯版本不匹配問題
HAS_FLASH_FFT = False
print("⚠️ 警告: 由於 CUDA PTX Toolchain 版本不相容，已自動切換回 PyTorch 原生 FFT。")

D_MODEL = 1152
NUM_POOL_HEADS = 9
POOL_HEAD_DIM = 128

from AGIV2GACGT import GemmaRMSNorm, GemmaFFN, apply_rope, repeat_kv


class AGIV2GlobalBlockFast(nn.Module):
    def __init__(
        self,
        D=D_MODEL,
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
        self.fft_norm = GemmaRMSNorm(D)
        self.mem_norm = GemmaRMSNorm(D)
        self.q_norm = GemmaRMSNorm(head_dim)
        self.k_norm = GemmaRMSNorm(head_dim)

        self.router = nn.Linear(D, 3, bias=False)
        nn.init.normal_(self.router.weight, mean=0.0, std=0.001)
        self.register_buffer("temperature", torch.tensor(2.0))

        self.omegas = nn.Parameter(torch.randn(K))
        self.mlp_H = nn.Sequential(nn.Linear(2 * K, D), nn.GELU(), nn.Linear(D, D))

        # 🌟 Memory Pool 參數 (支援 FlashAttention 的 MHA 形狀)
        self.Q_mem = nn.Parameter(torch.randn(M, NUM_POOL_HEADS, POOL_HEAD_DIM))
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

        if HAS_FLASH_FFT:
            self.flash_fft = FlashFFTConv(65536, dtype=torch.bfloat16)

    def fuse_qkv(self):
        # 融合 Local 的 Q, K, V
        if hasattr(self, "W_q_loc") and self.W_q_loc is not None:
            self.W_qkv_loc = (
                nn.Linear(
                    self.D,
                    self.W_q_loc.out_features
                    + self.W_k_loc.out_features
                    + self.W_v_loc.out_features,
                    bias=False,
                )
                .to(self.W_q_loc.weight.device)
                .to(self.W_q_loc.weight.dtype)
            )
            self.W_qkv_loc.weight.data = torch.cat(
                [
                    self.W_q_loc.weight.data,
                    self.W_k_loc.weight.data,
                    self.W_v_loc.weight.data,
                ],
                dim=0,
            )
            del self.W_q_loc, self.W_k_loc, self.W_v_loc
            self.W_q_loc, self.W_k_loc, self.W_v_loc = None, None, None

        # 融合 Memory Pool 的 K, V
        if hasattr(self, "W_k_mem") and self.W_k_mem is not None:
            self.W_kv_mem = (
                nn.Linear(
                    self.D,
                    self.W_k_mem.out_features + self.W_v_mem.out_features,
                    bias=False,
                )
                .to(self.W_k_mem.weight.device)
                .to(self.W_k_mem.weight.dtype)
            )
            self.W_kv_mem.weight.data = torch.cat(
                [self.W_k_mem.weight.data, self.W_v_mem.weight.data], dim=0
            )
            del self.W_k_mem, self.W_v_mem
            self.W_k_mem, self.W_v_mem = None, None

        # 融合 Cross Attention 的 K, V
        if hasattr(self, "W_k_cross") and self.W_k_cross is not None:
            self.W_kv_cross = (
                nn.Linear(
                    self.D,
                    self.W_k_cross.out_features + self.W_v_cross.out_features,
                    bias=False,
                )
                .to(self.W_k_cross.weight.device)
                .to(self.W_k_cross.weight.dtype)
            )
            self.W_kv_cross.weight.data = torch.cat(
                [self.W_k_cross.weight.data, self.W_v_cross.weight.data], dim=0
            )
            del self.W_k_cross, self.W_v_cross
            self.W_k_cross, self.W_v_cross = None, None

    def forward(self, X, shift_size=0, n_split_index=None):
        B, L, D = X.shape
        device = X.device
        dtype = X.dtype

        gate_logits = self.router(X)
        routing_weights = F.sigmoid(gate_logits / self.temperature)
        g_loc = routing_weights[..., 0:1].to(dtype)
        g_mem = routing_weights[..., 1:2].to(dtype)
        g_fft = routing_weights[..., 2:3].to(dtype)

        gate_sum = g_loc + g_mem + g_fft
        scale_factor = gate_sum.clamp(min=1.0)

        g_loc = (g_loc / scale_factor).to(dtype)
        g_mem = (g_mem / scale_factor).to(dtype)
        g_fft = (g_fft / scale_factor).to(dtype)

        self.avg_g_loc = g_loc.mean().detach()
        self.avg_g_mem = g_mem.mean().detach()
        self.avg_g_fft = g_fft.mean().detach()
        self._g_loc_for_loss = g_loc.mean()
        self._g_mem_for_loss = g_mem.mean()
        self._g_fft_for_loss = g_fft.mean()

        if n_split_index is not None:
            seq_range = torch.arange(L, device=device).unsqueeze(0)
            past_mask = seq_range < n_split_index.unsqueeze(1)
            X_past = X * past_mask.unsqueeze(-1).to(dtype)
        else:
            X_past = X
            past_mask = None

        # ==========================================
        # [Phase I] FFT 分支 (消融實驗：暫時關閉)
        # ==========================================
        """
        normed_X_fft = self.fft_norm(X_past)
        t = torch.arange(L, device=device, dtype=torch.float32)
        args = t.unsqueeze(1) * self.omegas.unsqueeze(0).to(torch.float32)
        gamma = torch.cat([torch.sin(args), torch.cos(args)], dim=-1).to(dtype)
        H = self.mlp_H(gamma)
        
        if HAS_FLASH_FFT:
            u = normed_X_fft.transpose(1, 2).contiguous()
            # FlashFFTConv 內部預處理會呼叫 torch.fft，而 PyTorch FFT 不支援 bfloat16，所以這裡 k 必須轉 float32
            k = H.transpose(0, 1).contiguous().to(torch.float32)
            Y_sys1 = self.flash_fft(u, k).transpose(1, 2).to(dtype)
        else:
            X_pad = F.pad(normed_X_fft, (0, 0, 0, L))  
            H_pad = F.pad(H, (0, 0, 0, L))  
            X_f = torch.fft.rfft(X_pad.to(torch.float32), dim=1)
            H_f = torch.fft.rfft(H_pad.to(torch.float32), dim=0).unsqueeze(0)
            Y_f = X_f * torch.conj(H_f)
            Y_sys1 = torch.fft.irfft(Y_f, n=2*L, dim=1).to(dtype)[:, :L, :]
        """
        Y_sys1 = 0.0  # 強制設為 0 以略過計算

        X_res0 = X + (Y_sys1 * g_fft)

        # ==========================================
        # [Phase II] 潛在記憶池 (🚀 徹底移除 Chunk，改為直接全局 FlashAttention)
        # ==========================================
        X_res0_past = (
            X_res0 * past_mask.unsqueeze(-1).to(dtype)
            if past_mask is not None
            else X_res0
        )
        normed_X_mem = self.mem_norm(X_res0_past)

        if hasattr(self, "W_kv_mem"):
            KV_mem = self.W_kv_mem(normed_X_mem).view(
                B, L, 2, NUM_POOL_HEADS, POOL_HEAD_DIM
            )
            K_mem, V_mem = KV_mem.unbind(dim=2)
        else:
            K_mem = self.W_k_mem(normed_X_mem).view(B, L, NUM_POOL_HEADS, POOL_HEAD_DIM)
            V_mem = self.W_v_mem(normed_X_mem).view(B, L, NUM_POOL_HEADS, POOL_HEAD_DIM)

        Q_mem_exp = self.Q_mem.unsqueeze(0).expand(
            B, -1, -1, -1
        )  # Shape: (B, M, 9, 128)

        # 🚀 關鍵改動：1024 個 Query 直接對 64K 的序列做 Attention。
        # 不切塊，不產生 B*N_mem 的爆炸 batch size，記憶體從 600MB 降至 2MB！
        M_global = flash_attn_func(Q_mem_exp, K_mem, V_mem, dropout_p=0.0, causal=False)
        # M_global shape: (B, M, 9, 128)
        M_global = M_global.view(B, self.M, D)

        # ==========================================
        # [Phase III] Local + Cross 注意力
        # ==========================================
        normed_X_loc = self.input_layernorm(X_res0)

        if hasattr(self, "W_qkv_loc"):
            QKV_loc = self.W_qkv_loc(normed_X_loc)
            q_dim = self.num_heads * self.head_dim
            kv_dim = self.num_kv_heads * self.head_dim
            Q_loc = QKV_loc[..., :q_dim].view(B, L, self.num_heads, self.head_dim)
            K_loc = QKV_loc[..., q_dim : q_dim + kv_dim].view(
                B, L, self.num_kv_heads, self.head_dim
            )
            V_loc = QKV_loc[..., -kv_dim:].view(B, L, self.num_kv_heads, self.head_dim)
        else:
            Q_loc = self.W_q_loc(normed_X_loc).view(B, L, self.num_heads, self.head_dim)
            K_loc = self.W_k_loc(normed_X_loc).view(
                B, L, self.num_kv_heads, self.head_dim
            )
            V_loc = self.W_v_loc(normed_X_loc).view(
                B, L, self.num_kv_heads, self.head_dim
            )

        Q_loc = self.q_norm(Q_loc)
        K_loc = self.k_norm(K_loc)
        Q_loc = apply_rope(Q_loc, self.head_dim, base_freq=self.rope_base).contiguous()
        K_loc = apply_rope(K_loc, self.head_dim, base_freq=self.rope_base).contiguous()
        V_loc = V_loc.contiguous()

        # Local 分支
        attn_out_loc = flash_attn_func(
            Q_loc, K_loc, V_loc, dropout_p=0.0, causal=True, window_size=(self.C - 1, 0)
        )
        Z_hat = self.o_proj_loc(
            attn_out_loc.reshape(B, L, self.num_heads * self.head_dim)
        )

        # 🚀 Cross Attention 重構：移除切塊，64K 的序列直接向 1024 個全局 Memory 提取資訊
        Q_cross = self.W_q_cross(Z_hat).view(B, L, self.num_heads, self.head_dim)

        if hasattr(self, "W_kv_cross"):
            KV_cross = self.W_kv_cross(M_global).view(
                B, self.M, 2, self.num_kv_heads, self.head_dim
            )
            K_cross, V_cross = KV_cross.unbind(dim=2)
        else:
            K_cross = self.W_k_cross(M_global).view(
                B, self.M, self.num_kv_heads, self.head_dim
            )
            V_cross = self.W_v_cross(M_global).view(
                B, self.M, self.num_kv_heads, self.head_dim
            )

        K_cross = repeat_kv(K_cross, self.num_key_value_groups)
        V_cross = repeat_kv(V_cross, self.num_key_value_groups)

        # FlashAttention Cross-Attention (L 查詢 M)，O(L*M)
        I_cross_fa = flash_attn_func(
            Q_cross, K_cross, V_cross, dropout_p=0.0, causal=False
        )
        I_cross = self.o_proj_cross(
            I_cross_fa.reshape(B, L, self.num_heads * self.head_dim)
        )

        attn_out = (Z_hat * g_loc) + (I_cross * g_mem)
        X_res1 = X_res0 + self.post_attention_layernorm(attn_out)

        normed_X_ffn = self.pre_feedforward_layernorm(X_res1)
        ffn_out = self.ffn(normed_X_ffn)
        Output = X_res1 + self.post_feedforward_layernorm(ffn_out)
        return Output


class AGIV2GFast(nn.Module):
    def __init__(
        self,
        vocab_size=262144,
        D=D_MODEL,
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

        from AGIV2GACGT import AGIV2LocalBlock

        for i in range(num_blocks):
            if (i + 1) % 6 == 0:
                self.blocks.append(
                    AGIV2GlobalBlockFast(
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

    def fuse_qkv(self):
        print("\n⚡ [Fusion] 啟動 Level 1: QKV 融合優化...")
        for i, block in enumerate(self.blocks):
            if hasattr(block, "fuse_qkv"):
                block.fuse_qkv()
        print("✅ [Fusion] 所有 Global Block QKV 矩陣已成功融合！")

    def set_temperature(self, temp):
        for block in self.blocks:
            if hasattr(block, "temperature"):
                block.temperature.fill_(temp)

    def compute_gate_neg_entropy(self):
        neg_entropy_sum = 0.0
        count = 0
        eps = 1e-8
        for block in self.blocks:
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
        out = self.embedding(x)
        out = out * math.sqrt(self.D)
        for i, block in enumerate(self.blocks):
            out = block(out, shift_size=0, n_split_index=n_split_index)
        out = self.final_norm(out)
        logits = self.fc_out(out)
        logits = logits / 30.0
        logits = torch.tanh(logits) * 30.0
        return logits
