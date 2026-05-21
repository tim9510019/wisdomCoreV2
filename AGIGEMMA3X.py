import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import math

# 🌟 官方 FlashAttention 導入 (帶有極為穩健的 CPU/無CUDA相容性 Fallback 機制)
try:
    from flash_attn import flash_attn_func
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False
    
    def flash_attn_func(q, k, v, dropout_p=0.0, causal=True, window_size=(-1, -1)):
        # 進行與 FlashAttention 行為完全等價的 PyTorch 標準自注意力原生實作 (支援 CPU/GPU 跨平台)
        q_t = q.transpose(1, 2)  # [B, num_heads, L, head_dim]
        k_t = k.transpose(1, 2)  # [B, num_kv_heads, L, head_dim]
        v_t = v.transpose(1, 2)  # [B, num_kv_heads, L, head_dim]
        
        # 若是 MQA/GQA，則重複 KV heads 以對齊 Query heads
        num_heads = q_t.shape[1]
        num_kv_heads = k_t.shape[1]
        if num_heads != num_kv_heads:
            n_rep = num_heads // num_kv_heads
            k_t = repeat_kv(k_t.transpose(1, 2), n_rep).transpose(1, 2)
            v_t = repeat_kv(v_t.transpose(1, 2), n_rep).transpose(1, 2)
            
        B, num_heads, L, head_dim = q_t.shape
        scores = torch.matmul(q_t, k_t.transpose(-2, -1)) / math.sqrt(head_dim)
        
        mask = torch.ones((L, L), dtype=torch.bool, device=q.device)
        if causal:
            mask = torch.tril(mask)
            
        if window_size != (-1, -1):
            C = window_size[0] + 1
            # 建立滑動窗口遮罩：限制 j 落在 [i - C + 1, i] 區間
            window_mask = torch.triu(torch.ones((L, L), dtype=torch.bool, device=q.device), diagonal=-(C - 1))
            mask = mask & window_mask
            
        scores = scores.masked_fill(~mask.unsqueeze(0).unsqueeze(1), float('-inf'))
        attn_weights = F.softmax(scores.to(torch.float32), dim=-1).to(q.dtype)
        out = torch.matmul(attn_weights, v_t)
        return out.transpose(1, 2).contiguous()


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
# 頻域縮放模組 (Spectral Downsampler & Upsampler)
# ==========================================
class SpectralDownsampler(nn.Module):
    """
    頻域理想下採樣模組 (Spectral Downsampler)
    在不破壞特徵能量密度的情況下，將序列長度從 L 理想降維至 L // factor
    """
    def __init__(self, factor=2):
        super().__init__()
        assert factor > 0 and (factor & (factor - 1)) == 0, "下採樣倍數必須是 2 的冪次方"
        self.factor = factor

    def forward(self, X, n_split_index=None):
        B, L, D = X.shape
        M = self.factor
        if M == 1:
            return X
            
        L_new = L // M
        device = X.device
        dtype = X.dtype

        # 🌟 因果隔離：若傳入 n_split_index，將未來區段抹零，防止未來資訊在 RFFT 轉換時發生全局洩漏
        if n_split_index is not None:
            seq_range = torch.arange(L, device=device).unsqueeze(0)
            past_mask = seq_range < n_split_index.unsqueeze(1)
            X_input = X * past_mask.unsqueeze(-1).to(dtype)
        else:
            X_input = X

        # 頻域轉換
        X_f = torch.fft.rfft(X_input.to(torch.float32), dim=1)  # [B, L // 2 + 1, D]
        
        # 理想低通濾波截斷
        X_f_down = X_f[:, : L_new // 2 + 1, :]                  # [B, L_new // 2 + 1, D]
        
        # 逆變換還原時域
        X_down = torch.fft.irfft(X_f_down, n=L_new, dim=1).to(dtype)
        
        # 能量振幅校正：除以 M 以保持均值與能量的一致性
        return X_down / M


class SpectralUpsampler(nn.Module):
    """
    頻域零填充上採樣模組 (Spectral Upsampler)
    利用 Sinc 理想重構插值，將序列長度從 L_down 恢復至 L_down * factor
    """
    def __init__(self, factor=2):
        super().__init__()
        assert factor > 0 and (factor & (factor - 1)) == 0, "上採樣倍數必須是 2 的冪次方"
        self.factor = factor

    def forward(self, X_down):
        M = self.factor
        if M == 1:
            return X_down
            
        B, L_down, D = X_down.shape
        L_new = L_down * M
        device = X_down.device
        dtype = X_down.dtype

        # 頻域轉換
        X_f_down = torch.fft.rfft(X_down.to(torch.float32), dim=1)  # [B, L_down // 2 + 1, D]
        
        # 高頻零填充 (Zero-Padding)
        pad_size = (L_new // 2 + 1) - (L_down // 2 + 1)
        # F.pad 從最後一個維度向左進行填充，我們填充頻域維度(dim=1)，對應 [B, F, D] 的倒數第二維
        X_f = F.pad(X_f_down, (0, 0, 0, pad_size))                 # [B, L_new // 2 + 1, D]
        
        # 逆變換還原時域
        X_up = torch.fft.irfft(X_f, n=L_new, dim=1).to(dtype)
        
        # 能量振幅校正：乘以 M 以保持重建信號的振幅規模
        return X_up * M


# ==========================================
# FFT 濾波器核心網路區塊
# ==========================================
class PureFFTBlock(nn.Module):
    def __init__(self, D=1152, K=1024):
        super().__init__()
        self.fft_norm = GemmaRMSNorm(D)
        self.omegas = nn.Parameter(torch.randn(K))
        self.mlp_H = nn.Sequential(nn.Linear(2 * K, D), nn.GELU(), nn.Linear(D, D))

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
        Y_sys1 = torch.fft.irfft(Y_f, n=2 * L, dim=1).to(dtype)[:, :L, :]

        Output = X + Y_sys1
        return Output


class DynamicPhaseLockFFTBlock(nn.Module):
    def __init__(self, D=1152, K=1024):
        super().__init__()
        self.fft_norm = GemmaRMSNorm(D)
        self.omegas = nn.Parameter(torch.randn(K))
        self.mlp_H = nn.Sequential(nn.Linear(2 * K, D), nn.GELU(), nn.Linear(D, D))
        self.eps = 1e-8

    def forward(self, X, shift_size=0, n_split_index=None):
        B, L, D = X.shape
        device = X.device
        dtype = X.dtype

        # 1. 因果隔離
        if n_split_index is not None:
            seq_range = torch.arange(L, device=device).unsqueeze(0)
            past_mask = seq_range < n_split_index.unsqueeze(1)
            X_past = X * past_mask.unsqueeze(-1).to(dtype)
        else:
            X_past = X

        normed_X_past = self.fft_norm(X_past)
        normed_X_full = self.fft_norm(X)

        # 2. 位置相關期望信號
        t = torch.arange(L, device=device, dtype=torch.float32)
        args = t.unsqueeze(1) * self.omegas.unsqueeze(0).to(torch.float32)
        gamma = torch.cat([torch.sin(args), torch.cos(args)], dim=-1).to(dtype)
        H = self.mlp_H(gamma)  # [L, D]

        # 3. 頻域動態相位鎖定
        S_f = torch.fft.rfft(normed_X_past.float(), dim=1)           # [B, L//2+1, D]
        H_f = torch.fft.rfft(H.float(), dim=0).unsqueeze(0)         # [1, L//2+1, D]
        V_f = torch.fft.rfft(normed_X_full.float(), dim=1)           # [B, L//2+1, D]

        # 互功率譜：過去信號 × 期望信號的共軛
        C_f = S_f * torch.conj(H_f)
        magnitude = torch.abs(C_f)

        # 軟性相位歸一化
        mean_energy = magnitude.mean(dim=1, keepdim=True)
        Phi_f = C_f / (magnitude + mean_energy + self.eps)

        # 將動態相位修正施加到完整序列
        Y_f = V_f * Phi_f
        Y = torch.fft.irfft(Y_f, n=L, dim=1).to(dtype)

        return X + Y


# ==========================================
# 完美對齊 Gemma3 Decoder Layer (降維運行版本)
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
        downsample_factor=1,  # 支援傳入降維倍率以動態縮放注意力窗口長度
    ):
        super().__init__()
        self.C = C
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_key_value_groups = num_heads // num_kv_heads
        self.is_global = is_global
        self.rope_base = rope_base
        self.downsample_factor = downsample_factor

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

        # 根據是否為 Global Attention 與降維倍率設定 Window Size
        if self.is_global:
            window_size = (-1, -1)
        else:
            # 滑動窗口隨著序列降維而同步收縮，維持物理比例一致
            scaled_C = max(self.C // self.downsample_factor, 1)
            window_size = (scaled_C - 1, 0)

        attn_out = flash_attn_func(
            Q, K, V, dropout_p=0.0, causal=True, window_size=window_size
        )

        Z_hat = self.o_proj_loc(attn_out.reshape(B, L, self.num_heads * self.head_dim))

        X_res1 = X + self.post_attention_layernorm(Z_hat)
        normed_X_ffn = self.pre_feedforward_layernorm(X_res1)
        ffn_out = self.ffn(normed_X_ffn)
        Output = X_res1 + self.post_feedforward_layernorm(ffn_out)
        return Output


# ==========================================
# AGIGEMMA3X 骨幹網路
# ==========================================
class AGIGEMMA3X(nn.Module):
    def __init__(
        self,
        vocab_size=262144,
        D=1152,
        hidden_dim=6912,
        num_blocks=26,
        N_fft_F=1,                  # 前置 FFT 層數
        N_fft_B=4,                  # 後置 FFT 層數 (預設 4 層負責重建)
        C=1024,
        K=1024,
        num_heads=4,
        num_kv_heads=1,
        head_dim=256,
        rope_local=10000.0,
        rope_global=1000000.0,
        fft_block_type="pure",      # "pure" 或 "dynamic"
        downsample_factor=2,        # 🌟 配置下採樣倍率 (必須是 2 的倍數，如 2, 4, 8)
    ):
        super().__init__()
        self.D = D
        self.downsample_factor = downsample_factor
        self.embedding = nn.Embedding(vocab_size, D)

        # 選擇 FFT Block 類型
        if fft_block_type == "dynamic":
            _make_fft = lambda: DynamicPhaseLockFFTBlock(D=D, K=K)
        else:
            _make_fft = lambda: PureFFTBlock(D=D, K=K)

        # 前置 FFT 模組
        self.pre_fft_blocks = nn.ModuleList(
            [_make_fft() for _ in range(N_fft_F)]
        )

        # 🌟 頻域縮放模組
        self.downsampler = SpectralDownsampler(factor=downsample_factor)
        self.upsampler = SpectralUpsampler(factor=downsample_factor)

        # Gemma3 主體 Block 組群
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
                    downsample_factor=downsample_factor,  # 穿透降維參數
                )
            )

        # 後置 FFT 模組 (負責把長度與特徵重建)
        self.post_fft_blocks = nn.ModuleList(
            [_make_fft() for _ in range(N_fft_B)]
        )

        self.final_norm = GemmaRMSNorm(D)
        self.fc_out = nn.Linear(D, vocab_size, bias=False)

    def forward(self, x, n_split_index=None):
        # 1. 詞向量與前置 FFT
        out = self.embedding(x)
        out = out * math.sqrt(self.D)

        for block in self.pre_fft_blocks:
            out = block(out, shift_size=0, n_split_index=n_split_index)

        # 2. 頻域降維
        out_down = self.downsampler(out, n_split_index=n_split_index)

        # 3. 因果分隔線按比例縮放
        if n_split_index is not None:
            n_split_index_down = n_split_index // self.downsample_factor
        else:
            n_split_index_down = None

        # 4. Gemma3 骨幹 (完全在降維空間中極速運行)
        for block in self.blocks:
            out_down = block(out_down, shift_size=0, n_split_index=n_split_index_down)

        # 5. 頻域升維 (Sinc 完美插值還原長度)
        out = self.upsampler(out_down)

        # 6. 後置 FFT
        for block in self.post_fft_blocks:
            out = block(out, shift_size=0, n_split_index=n_split_index)

        # 7. Final Norm 與極致飽和 Logit
        out = self.final_norm(out)
        logits = self.fc_out(out)
        logits = logits / 30.0
        logits = torch.tanh(logits) * 30.0
        return logits


# ==========================================
# AGIGEMMA3X 專屬 CausalLM 訓練/推論包裝器
# ==========================================
class AGIGEMMA3XForCausalLM(nn.Module):
    def __init__(self, base_model, use_gc=True):
        super().__init__()
        self.base_model = base_model
        self.use_gc = use_gc

    def forward(self, input_ids, labels=None, **kwargs):
        n_split_index = kwargs.get("n_split_index", None)

        # 1. 前置嵌入特徵
        hidden_states = self.base_model.embedding(input_ids)
        hidden_states = hidden_states * math.sqrt(self.base_model.D)

        # 2. 執行前置外層 FFT
        for block in self.base_model.pre_fft_blocks:
            if self.training and self.use_gc:
                hidden_states = checkpoint.checkpoint(
                    block, hidden_states, 0, n_split_index, use_reentrant=False
                )
            else:
                hidden_states = block(
                    hidden_states, shift_size=0, n_split_index=n_split_index
                )

        # 3. 頻域下採樣降維 (時域序列 L -> L // M)
        hidden_states_down = self.base_model.downsampler(hidden_states, n_split_index=n_split_index)

        # 4. 縮放因果隔離位置
        if n_split_index is not None:
            n_split_index_down = n_split_index // self.base_model.downsample_factor
        else:
            n_split_index_down = None

        # 5. 穿透凍結的 26 層 Gemma3 骨幹 (以 L // M 長度在時域低算力空間狂飆)
        for i, block in enumerate(self.base_model.blocks):
            shift_size = (block.C // 2) if (i % 2 != 0) else 0
            shift_size_down = shift_size // self.base_model.downsample_factor
            if self.training and self.use_gc:
                hidden_states_down = checkpoint.checkpoint(
                    block, hidden_states_down, shift_size_down, n_split_index_down, use_reentrant=False
                )
            else:
                hidden_states_down = block(
                    hidden_states_down, shift_size=shift_size_down, n_split_index=n_split_index_down
                )

        # 6. 頻域上採樣升維 (時域序列 L // M -> L)
        hidden_states = self.base_model.upsampler(hidden_states_down)

        # 7. 穿透後置 4 層外層 FFT
        for block in self.base_model.post_fft_blocks:
            if self.training and self.use_gc:
                hidden_states = checkpoint.checkpoint(
                    block, hidden_states, 0, n_split_index, use_reentrant=False
                )
            else:
                hidden_states = block(
                    hidden_states, shift_size=0, n_split_index=n_split_index
                )

        hidden_states = self.base_model.final_norm(hidden_states)

        loss = None
        logits = None

        # 8. 計算損失 (分塊以防大序列記憶體溢出)
        if labels is not None:
            shift_hidden = hidden_states[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss_fct = nn.CrossEntropyLoss(reduction="sum", ignore_index=-100)
            total_loss = 0.0
            valid_tokens = 0.0
            chunk_size = 2048
            seq_len = shift_hidden.size(1)

            for i in range(0, seq_len, chunk_size):
                end_idx = min(i + chunk_size, seq_len)
                c_hidden = shift_hidden[:, i:end_idx, :]
                c_logits = self.base_model.fc_out(c_hidden)
                c_labels = shift_labels[:, i:end_idx]

                c_loss = loss_fct(
                    c_logits.reshape(-1, c_logits.size(-1)).float(),
                    c_labels.reshape(-1),
                )
                total_loss += c_loss

                valid_tokens += (c_labels != -100).sum().item()
                del c_logits, c_hidden

            loss = total_loss / max(valid_tokens, 1)

            if self.training:
                self._last_ce_loss = loss.detach().item()
        else:
            logits = self.base_model.fc_out(hidden_states)

        return (
            {"loss": loss, "logits": logits} if logits is not None else {"loss": loss}
        )
