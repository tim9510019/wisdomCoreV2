import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def apply_rope(x):
    """
    對輸入 x 套用 RoPE。
    x shape: (B, N, C_len, D_inner) 
    """
    B, N, C_len, D_inner = x.shape
    position = torch.arange(C_len, device=x.device).unsqueeze(1).float()
    div_term = torch.exp(torch.arange(0, D_inner, 2, device=x.device).float() * -(math.log(10000.0) / D_inner))
    freqs = position * div_term 
    
    freqs = freqs.unsqueeze(0).unsqueeze(0) 
    sin_val = torch.sin(freqs)
    cos_val = torch.cos(freqs)
    
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    x_rot = torch.empty_like(x)
    x_rot[..., 0::2] = x1 * cos_val - x2 * sin_val
    x_rot[..., 1::2] = x1 * sin_val + x2 * cos_val
    return x_rot

class AGIBlock(nn.Module):
    def __init__(self, D, D_inner, K, M, C, hidden_dim):
        super().__init__()
        self.D = D
        self.D_inner = D_inner
        self.K = K
        self.M = M
        self.C = C
        
        # Phase I: 傅立葉共振 (FFM)
        self.omegas = nn.Parameter(torch.randn(K))
        self.mlp_H = nn.Sequential(
            nn.Linear(2 * K, D_inner),
            nn.GELU(),
            nn.Linear(D_inner, D)
        )
        self.norm1 = nn.LayerNorm(D)
        
        # Phase II: 潛在記憶池 (Latent Memory)
        self.Q_mem = nn.Parameter(torch.randn(M, D_inner))
        self.W_k_mem = nn.Linear(D, D_inner, bias=False)
        self.W_v_mem = nn.Linear(D, D, bias=False)
        
        # Phase III: 空間路由 (Local & Cross Attention)
        self.W_q_loc = nn.Linear(D, D_inner, bias=False)
        self.W_k_loc = nn.Linear(D, D_inner, bias=False)
        self.W_v_loc = nn.Linear(D, D, bias=False)
        
        self.W_q_cross = nn.Linear(D, D_inner, bias=False)
        self.W_k_cross = nn.Linear(D, D_inner, bias=False)
        self.W_v_cross = nn.Linear(D, D, bias=False)
        
        self.norm2 = nn.LayerNorm(D)
        self.ffn = nn.Sequential(
            nn.Linear(D, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, D)
        )

    def forward(self, X):
        B, L, D = X.shape
        device = X.device
        
        # ==================== Phase I ====================
        t = torch.arange(L, device=device, dtype=torch.float)
        args = t.unsqueeze(1) * self.omegas.unsqueeze(0) 
        gamma = torch.cat([torch.sin(args), torch.cos(args)], dim=-1) 
        H = self.mlp_H(gamma)
        
        X_pad = F.pad(X, (0, 0, 0, L))  
        H_pad = F.pad(H, (0, 0, 0, L))  
        
        X_f = torch.fft.rfft(X_pad, dim=1)
        H_f = torch.fft.rfft(H_pad, dim=0).unsqueeze(0) 
        
        Y_sys1_f = X_f * torch.conj(H_f)
        Y_sys1_pad = torch.fft.irfft(Y_sys1_f, n=2*L, dim=1)
        Y_sys1 = Y_sys1_pad[:, :L, :] 
        Z = self.norm1(X + Y_sys1) 
        
        # ==================== Phase II ====================
        K_mem = self.W_k_mem(Z) 
        V_mem = self.W_v_mem(Z) 
        
        Q_mem_b = self.Q_mem.unsqueeze(0).expand(B, -1, -1) 
        attn_mem = torch.bmm(Q_mem_b, K_mem.transpose(1, 2)) / math.sqrt(self.D_inner)
        attn_mem_probs = F.softmax(attn_mem, dim=-1)
        M_global = torch.bmm(attn_mem_probs, V_mem) 
        
        # ==================== Phase III ====================
        pad_len = (self.C - L % self.C) % self.C
        if pad_len > 0:
            Z = F.pad(Z, (0, 0, 0, pad_len))
        L_pad = L + pad_len
        N = L_pad // self.C
        
        Z_chunked = Z.view(B, N, self.C, D) 
        
        Q_loc = self.W_q_loc(Z_chunked) 
        K_loc = self.W_k_loc(Z_chunked) 
        V_loc = self.W_v_loc(Z_chunked) 
        
        Q_loc = apply_rope(Q_loc)
        K_loc = apply_rope(K_loc)
        
        attn_loc = torch.matmul(Q_loc, K_loc.transpose(-1, -2)) / math.sqrt(self.D_inner)
        Z_hat = Z_chunked + torch.matmul(F.softmax(attn_loc, dim=-1), V_loc) 
        
        Q_cross = self.W_q_cross(Z_hat) 
        K_cross = self.W_k_cross(M_global).unsqueeze(1) 
        V_cross = self.W_v_cross(M_global).unsqueeze(1) 
        
        attn_cross = torch.matmul(Q_cross, K_cross.transpose(-1, -2)) / math.sqrt(self.D_inner)
        I_cross = torch.matmul(F.softmax(attn_cross, dim=-1), V_cross) 
        
        Z_tilde_chunked = self.norm2(Z_hat + I_cross) 
        Z_tilde = Z_tilde_chunked.view(B, L_pad, D)
        
        if pad_len > 0:
            Z_tilde = Z_tilde[:, :L, :]  
            
        Output = Z_tilde + self.ffn(Z_tilde) 
        return Output

class HolographicMemoryNetwork(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_blocks, DI=64, K=64, M=64, C=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # 動態降維：瓶頸維度為 embedding 的一半，維持 O(D) 的高效運算
        D_inner = DI
        
        self.blocks = nn.ModuleList([
            AGIBlock(embed_dim, D_inner, K, M, C, hidden_dim) 
            for _ in range(num_blocks)
        ])
        self.fc_out = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        out = self.embedding(x)
        for block in self.blocks:
            out = block(out)
        return self.fc_out(out)