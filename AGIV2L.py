import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ==========================================
# 基礎 SOTA 元件庫 (對齊 LLaMA-4 與硬體加速)
# ==========================================

class RMSNorm(nn.Module):
    """SOTA 標配：Pre-Norm 穩定器"""
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # 強制在 float32 計算，確保運算過程不溢位且保留精度
        norm_x = x.to(torch.float32)
        norm = torch.rsqrt(norm_x.pow(2).mean(-1, keepdim=True) + self.eps)
        # 修正：在 float32 空間完成相乘後再轉換回 bf16/fp16
        return ((norm_x * norm).to(x.dtype)) * self.weight

class SwiGLUFFN(nn.Module):
    """LLaMA-4 知識庫的物理載體"""
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

def apply_rope(x, head_dim):
    """
    局部旋轉位置編碼 (只作用於 D_head)
    注意：此處依原設計實作「窗口內相對位置」，即每個 C_len 獨立從 0 開始編碼。
    x shape: (B, N, num_heads, C_len, head_dim)
    """
    B, N, num_heads, C_len, _ = x.shape
    position = torch.arange(C_len, device=x.device).unsqueeze(1).float()
    div_term = torch.exp(torch.arange(0, head_dim, 2, device=x.device).float() * -(math.log(10000.0) / head_dim))
    freqs = position * div_term 
    
    # 擴展維度以對齊 x 
    freqs = freqs.unsqueeze(0).unsqueeze(0).unsqueeze(0) 
    sin_val = torch.sin(freqs)
    cos_val = torch.cos(freqs)
    
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    x_rot = torch.empty_like(x)
    x_rot[..., 0::2] = x1 * cos_val - x2 * sin_val
    x_rot[..., 1::2] = x1 * sin_val + x2 * cos_val
    return x_rot

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """GQA 擴展函數：將 KV heads 複製展開以對齊 Q heads"""
    B, N, num_kv_heads, seq_len, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, :, None, :, :].expand(B, N, num_kv_heads, n_rep, seq_len, head_dim)
    # 使用 reshape 確保安全
    return hidden_states.reshape(B, N, num_kv_heads * n_rep, seq_len, head_dim)

# ==========================================
# 核心網路區塊 (1:3 稀疏架構分化)
# ==========================================

class AGIV2LocalBlock(nn.Module):
    """
    Type A (輕量局部層)：
    負責短程文法解析與 Shift 資訊流傳遞。
    僅包含 Local SDPA 與 FFN。
    """
    def __init__(self, D=4096, hidden_dim=14336, C=1024, num_heads=32, num_kv_heads=8):
        super().__init__()
        self.C = C
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = D // num_heads
        self.num_key_value_groups = num_heads // num_kv_heads
        
        self.norm1 = RMSNorm(D)
        self.norm2 = RMSNorm(D)
        
        # Spatial Routing (Local GQA)
        self.W_q_loc = nn.Linear(D, num_heads * self.head_dim, bias=False)
        self.W_k_loc = nn.Linear(D, num_kv_heads * self.head_dim, bias=False)
        self.W_v_loc = nn.Linear(D, num_kv_heads * self.head_dim, bias=False)
        self.o_proj_loc = nn.Linear(num_heads * self.head_dim, D, bias=False)
        
        self.ffn = SwiGLUFFN(hidden_size=D, intermediate_size=hidden_dim)

    def forward(self, X, shift_size=0):
        B, L, D = X.shape
        device = X.device
        
        # Pre-Norm
        normed_X = self.norm1(X)
        
        # ===== Phase III (局部空間路由) =====
        pad_len = (self.C - L % self.C) % self.C
        Z = F.pad(normed_X, (0, 0, 0, pad_len)) if pad_len > 0 else normed_X
        L_pad = L + pad_len
        N = L_pad // self.C
        
        # 平移與布林遮罩
        if shift_size > 0:
            Z_shifted = torch.roll(Z, shifts=-shift_size, dims=1)
            region_idx = torch.zeros(L_pad, device=device)
            region_idx[-shift_size:] = 1 
            mask_windows = region_idx.view(1, N, 1, self.C, 1)
            sdpa_mask = (mask_windows == mask_windows.transpose(-1, -2)) 
        else:
            Z_shifted = Z
            sdpa_mask = None
            
        # 修正：使用 reshape 吸收 torch.roll 帶來的不連續性
        Z_chunked = Z_shifted.reshape(B, N, self.C, D)
        
        # GQA 投影與形狀轉換 (W投影後連續，可用view)
        Q_loc = self.W_q_loc(Z_chunked).view(B, N, self.C, self.num_heads, self.head_dim).transpose(2, 3)
        K_loc = self.W_k_loc(Z_chunked).view(B, N, self.C, self.num_kv_heads, self.head_dim).transpose(2, 3)
        V_loc = self.W_v_loc(Z_chunked).view(B, N, self.C, self.num_kv_heads, self.head_dim).transpose(2, 3)
        
        Q_loc = apply_rope(Q_loc, self.head_dim)
        K_loc = apply_rope(K_loc, self.head_dim)
        
        K_loc = repeat_kv(K_loc, self.num_key_value_groups)
        V_loc = repeat_kv(V_loc, self.num_key_value_groups)
        
        # 備註：若傳入 sdpa_mask，會自動降級為 Math Backend 以確保數學正確性
        Z_hat_chunked = F.scaled_dot_product_attention(
            Q_loc, K_loc, V_loc, 
            attn_mask=sdpa_mask, 
            dropout_p=0.0, 
            is_causal=False
        )
        
        # 修正：transpose 後重塑為 4D
        Z_hat_chunked = Z_hat_chunked.transpose(2, 3).reshape(B, N, self.C, D)
        Z_hat_chunked = self.o_proj_loc(Z_hat_chunked)
        
        Z_hat_shifted = Z_hat_chunked.reshape(B, L_pad, D)
        Z_hat = torch.roll(Z_hat_shifted, shifts=shift_size, dims=1) if shift_size > 0 else Z_hat_shifted
        
        if pad_len > 0:
            Z_hat = Z_hat[:, :L, :]
            
        # 殘差 1
        X_res1 = X + Z_hat
        
        # ===== FFN =====
        Output = X_res1 + self.ffn(self.norm2(X_res1))
        return Output

class AGIV2GlobalBlock(nn.Module):
    """
    Type B (重裝全局層)：
    負責長文本共振、潛在記憶池壓縮與檢索。
    包含 Phase I, II, III (Local+Cross) 與 FFN。
    """
    def __init__(self, D=4096, hidden_dim=14336, K=1024, M=1024, C=1024, num_heads=32, num_kv_heads=8):
        super().__init__()
        self.D = D
        self.C = C
        self.M = M
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = D // num_heads
        self.num_key_value_groups = num_heads // num_kv_heads
        
        self.norm1 = RMSNorm(D)
        self.norm2 = RMSNorm(D)
        self.norm3 = RMSNorm(D)
        
        # Phase I: 傅立葉共振
        self.omegas = nn.Parameter(torch.randn(K))
        self.mlp_H = nn.Sequential(
            nn.Linear(2 * K, D),
            nn.GELU(),
            nn.Linear(D, D)
        )
        
        # Phase II: 潛在記憶池
        self.Q_mem = nn.Parameter(torch.randn(M, D))
        self.W_k_mem = nn.Linear(D, D, bias=False)
        self.W_v_mem = nn.Linear(D, D, bias=False)
        
        # Phase III: 空間路由 (Local)
        self.W_q_loc = nn.Linear(D, num_heads * self.head_dim, bias=False)
        self.W_k_loc = nn.Linear(D, num_kv_heads * self.head_dim, bias=False)
        self.W_v_loc = nn.Linear(D, num_kv_heads * self.head_dim, bias=False)
        self.o_proj_loc = nn.Linear(num_heads * self.head_dim, D, bias=False)
        
        # Phase III: 空間路由 (Cross)
        self.W_q_cross = nn.Linear(D, num_heads * self.head_dim, bias=False)
        self.W_k_cross = nn.Linear(D, num_kv_heads * self.head_dim, bias=False)
        self.W_v_cross = nn.Linear(D, num_kv_heads * self.head_dim, bias=False)
        self.o_proj_cross = nn.Linear(num_heads * self.head_dim, D, bias=False)
        
        self.ffn = SwiGLUFFN(hidden_size=D, intermediate_size=hidden_dim)

    def forward(self, X, shift_size=0):
        B, L, D = X.shape
        device = X.device
        dtype = X.dtype
        
        # ===== Phase I (頻域共振) =====
        normed_X1 = self.norm1(X)
        
        t = torch.arange(L, device=device, dtype=torch.float32)
        args = t.unsqueeze(1) * self.omegas.unsqueeze(0).to(torch.float32)
        gamma = torch.cat([torch.sin(args), torch.cos(args)], dim=-1).to(dtype)
        H = self.mlp_H(gamma)
        
        X_pad = F.pad(normed_X1, (0, 0, 0, L))  
        H_pad = F.pad(H, (0, 0, 0, L))  
        
        # FFT 必須在 float32 下進行
        X_f = torch.fft.rfft(X_pad.to(torch.float32), dim=1)
        H_f = torch.fft.rfft(H_pad.to(torch.float32), dim=0).unsqueeze(0) 
        
        Y_sys1_f = X_f * torch.conj(H_f)
        Y_sys1_pad = torch.fft.irfft(Y_sys1_f, n=2*L, dim=1).to(dtype)
        Y_sys1 = Y_sys1_pad[:, :L, :] 
        
        X_res1 = X + Y_sys1
        
        # ===== Phase II & Phase III 共用 Pre-Norm =====
        normed_X2 = self.norm2(X_res1)
        
        # [Phase II] 潛在記憶池
        # 警告觀測：此處的 SDPA head_dim = D = 4096，將觸發 PyTorch Math Backend。
        K_mem = self.W_k_mem(normed_X2) 
        V_mem = self.W_v_mem(normed_X2) 
        Q_mem_b = self.Q_mem.unsqueeze(0).expand(B, -1, -1) 
        
        M_global = F.scaled_dot_product_attention(
            Q_mem_b.unsqueeze(1), 
            K_mem.unsqueeze(1), 
            V_mem.unsqueeze(1), 
            dropout_p=0.0, 
            is_causal=False
        ).squeeze(1) 
        
        # [Phase III] Local 空間路由
        pad_len = (self.C - L % self.C) % self.C
        Z = F.pad(normed_X2, (0, 0, 0, pad_len)) if pad_len > 0 else normed_X2
        L_pad = L + pad_len
        N = L_pad // self.C
        
        if shift_size > 0:
            Z_shifted = torch.roll(Z, shifts=-shift_size, dims=1)
            region_idx = torch.zeros(L_pad, device=device)
            region_idx[-shift_size:] = 1 
            mask_windows = region_idx.view(1, N, 1, self.C, 1)
            sdpa_mask = (mask_windows == mask_windows.transpose(-1, -2))
        else:
            Z_shifted = Z
            sdpa_mask = None
            
        # 修正：使用 reshape 處理可能的不連續性
        Z_chunked = Z_shifted.reshape(B, N, self.C, D) 
        
        Q_loc = self.W_q_loc(Z_chunked).view(B, N, self.C, self.num_heads, self.head_dim).transpose(2, 3)
        K_loc = self.W_k_loc(Z_chunked).view(B, N, self.C, self.num_kv_heads, self.head_dim).transpose(2, 3)
        V_loc = self.W_v_loc(Z_chunked).view(B, N, self.C, self.num_kv_heads, self.head_dim).transpose(2, 3)
        
        Q_loc = apply_rope(Q_loc, self.head_dim)
        K_loc = apply_rope(K_loc, self.head_dim)
        
        K_loc = repeat_kv(K_loc, self.num_key_value_groups)
        V_loc = repeat_kv(V_loc, self.num_key_value_groups)
        
        Z_hat_chunked = F.scaled_dot_product_attention(
            Q_loc, K_loc, V_loc, 
            attn_mask=sdpa_mask, 
            dropout_p=0.0, 
            is_causal=False
        )
        
        # 修正：transpose 後的重塑
        Z_hat_chunked = Z_hat_chunked.transpose(2, 3).reshape(B, N, self.C, D)
        Z_hat_chunked = self.o_proj_loc(Z_hat_chunked)
        
        Z_hat_shifted = Z_hat_chunked.reshape(B, L_pad, D)
        Z_hat = torch.roll(Z_hat_shifted, shifts=shift_size, dims=1) if shift_size > 0 else Z_hat_shifted
        
        # [Phase III] Cross 全局檢索路由
        Q_cross = self.W_q_cross(Z_hat).view(B, L_pad, self.num_heads, self.head_dim).transpose(1, 2)
        K_cross = self.W_k_cross(M_global).view(B, self.M, self.num_kv_heads, self.head_dim).transpose(1, 2)
        V_cross = self.W_v_cross(M_global).view(B, self.M, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        K_cross = repeat_kv(K_cross.unsqueeze(1), self.num_key_value_groups).squeeze(1)
        V_cross = repeat_kv(V_cross.unsqueeze(1), self.num_key_value_groups).squeeze(1)
        
        I_cross = F.scaled_dot_product_attention(
            Q_cross, K_cross, V_cross, 
            dropout_p=0.0, 
            is_causal=False
        )
        
        # 修正：安全重塑回 [B, L_pad, D]
        I_cross = I_cross.transpose(1, 2).reshape(B, L_pad, D)
        I_cross = self.o_proj_cross(I_cross)
        
        if pad_len > 0:
            Z_hat = Z_hat[:, :L, :]
            I_cross = I_cross[:, :L, :]
            
        X_res2 = X_res1 + Z_hat + I_cross
        
        # ===== FFN =====
        Output = X_res2 + self.ffn(self.norm3(X_res2))
        return Output

# ==========================================
# 主模型裝配
# ==========================================

class AGIV2L(nn.Module):
    """
    AGIV2L (Large) - 支援 32K~128K 超長上下文，融合 LLaMA-4 權重。
    採用 1:3 交錯稀疏配置，總層數 32，其中 8 層為 Global Block，24 層為 Local Block。
    """
    def __init__(self, vocab_size=128256, D=4096, hidden_dim=14336, num_blocks=32, 
                 C=1024, K=1024, M=1024):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, D)
        self.blocks = nn.ModuleList()
        
        for i in range(num_blocks):
            if (i + 1) % 4 == 0:
                self.blocks.append(
                    AGIV2GlobalBlock(D=D, hidden_dim=hidden_dim, K=K, M=M, C=C)
                )
            else:
                self.blocks.append(
                    AGIV2LocalBlock(D=D, hidden_dim=hidden_dim, C=C)
                )
                
        self.final_norm = RMSNorm(D)
        self.fc_out = nn.Linear(D, vocab_size, bias=False)

    def forward(self, x):
        out = self.embedding(x)
        for i, block in enumerate(self.blocks):
            shift_size = (block.C // 2) if (i % 2 != 0) else 0
            out = block(out, shift_size=shift_size)
        out = self.final_norm(out)
        return self.fc_out(out)