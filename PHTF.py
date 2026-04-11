import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# =================================================================
# PHTF — Phase-locked Holographic Transformer Fusion (零參數動態鎖相版)
# =================================================================
# 核心理念：融合 Transformer 的「精確注意力」與 PLHN 的「FFT 頻域分析」
# 突破更新：
# 1. Asymmetric Shared-QK + RoPE：以 O(D) 成本打破對稱性枷鎖。
# 2. DynamicHologram：零新增參數，直接利用 Attention 分支的 QK 與 V，
#    透過頻域互相關 FFT(QK)*conj(FFT(V)) 萃取動態相位差，實現資料驅動的波形對齊。
# =================================================================

MAX_SEQ_LEN = 8192
SEQ_LEN = 64
BATCH_SIZE = 16
EMBED_DIM = 128
NUM_HEADS = 4
HIDDEN_DIM = 256
VOCAB_SIZE = 128
NUM_EPOCHS = 20

class CustomDataset(Dataset):
    def __init__(self, num_samples=5000, seq_len=SEQ_LEN):
        self.num_samples = num_samples
        self.seq_len = seq_len
        
        self.data_x = torch.zeros((self.num_samples, self.seq_len), dtype=torch.long)
        self.data_y = torch.zeros((self.num_samples,), dtype=torch.long)
        
        for i in range(self.num_samples):
            seq = torch.randint(0, 128, (self.seq_len - 1,))
            key_idx = torch.randint(0, self.seq_len - 2, (1,)).item()
            key = seq[key_idx]
            target = seq[key_idx + 1]
            
            self.data_x[i, :self.seq_len - 1] = seq
            self.data_x[i, self.seq_len - 1] = key
            self.data_y[i] = target

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data_x[idx], self.data_y[idx]

# =========================================================
# RoPE (Rotary Position Embedding) 輔助函數
# =========================================================
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """預先計算 RoPE 需要的旋轉頻率 (cos, sin)"""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cos(freqs)
    freqs_sin = torch.sin(freqs)
    return freqs_cos, freqs_sin

def apply_rotary_emb(xq, xk, freqs_cos, freqs_sin):
    """將 RoPE 套用到 Q 和 K 上"""
    xq_r, xq_i = xq.float().reshape(xq.shape[:-1] + (-1, 2)).unbind(-1)
    xk_r, xk_i = xk.float().reshape(xk.shape[:-1] + (-1, 2)).unbind(-1)
    
    freqs_cos = freqs_cos.view(1, xq.shape[1], 1, -1)
    freqs_sin = freqs_sin.view(1, xq.shape[1], 1, -1)
    
    xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
    xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos
    xq_out = torch.stack([xq_out_r, xq_out_i], dim=-1).flatten(3)
    
    xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin
    xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos
    xk_out = torch.stack([xk_out_r, xk_out_i], dim=-1).flatten(3)
    
    return xq_out.type_as(xq), xk_out.type_as(xk)

# =========================================================
# 子模組 1：Asymmetric Shared-Projection Attention
# =========================================================
class AsymmetricSharedAttention(nn.Module):
    """結合 RoPE 與輕量級不對稱擾動的 Shared-QK Attention，並支援預計算特徵注入"""
    def __init__(self, embed_dim, num_heads, max_seq_len=512):
        super(AsymmetricSharedAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim

        self.qk_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.out_linear = nn.Linear(embed_dim, embed_dim)
        
        # O(D) 維度的可學習縮放向量，打破對稱性
        self.k_scale = nn.Parameter(torch.ones(embed_dim))

        # 註冊 RoPE buffer
        freqs_cos, freqs_sin = precompute_freqs_cis(self.head_dim, max_seq_len)
        self.register_buffer("freqs_cos", freqs_cos)
        self.register_buffer("freqs_sin", freqs_sin)

    def forward_with_precomputed(self, x, shared_qk, V):
        """接收外部已經算好的 shared_qk 和 V，避免與 Hologram 重複計算"""
        batch_size, seq_len, embed_dim = x.size()

        # 產生不對稱的 Q 與 K
        Q = shared_qk
        K = shared_qk * self.k_scale 

        Q_split = Q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        K_split = K.view(batch_size, seq_len, self.num_heads, self.head_dim)
        V_split = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 套用 RoPE
        f_cos = self.freqs_cos[:seq_len]
        f_sin = self.freqs_sin[:seq_len]
        Q_rot, K_rot = apply_rotary_emb(Q_split, K_split, f_cos, f_sin)

        Q_rot = Q_rot.transpose(1, 2)
        K_rot = K_rot.transpose(1, 2)

        # 計算不對稱的注意力分數
        scores = torch.matmul(Q_rot, K_rot.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = torch.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V_split)

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        out = self.out_linear(attn_output)
        
        return out

    def forward(self, x):
        """標準呼叫介面，如果單獨使用此模組時"""
        shared_qk = self.qk_linear(x)
        V = self.v_linear(x)
        return self.forward_with_precomputed(x, shared_qk, V)

# =========================================================
# 子模組 2：Dynamic Hologram (零參數動態鎖相)
# =========================================================
class DynamicHologram(nn.Module):
    """
    零新增參數的動態鎖相模組。
    直接接收 Attention 分支算好的 QK 與 V，
    萃取預期(QK)與內容(V)之間的動態時間錯位，並施加於 V 上。
    複雜度維持在 O(N log N)。
    """
    def __init__(self):
        super(DynamicHologram, self).__init__()
        self.eps = 1e-8

    def forward(self, qk_tensor, v_tensor):
        # 1. 將 QK 與 V 轉入頻域
        QK_f = torch.fft.rfft(qk_tensor, dim=1)
        V_f = torch.fft.rfft(v_tensor, dim=1)
        
        # 2. 頻域互相關：計算 QK 與 V 的互功率譜
        cross_power_spectrum = QK_f * torch.conj(V_f)
        magnitude = torch.abs(cross_power_spectrum)
        
        # 3. 突破點：計算當下特徵維度在序列上的「平均頻譜能量」
        # 對頻率維度 (dim=1) 取平均，保持維度以便廣播 (Broadcast)
        mean_energy = magnitude.mean(dim=1, keepdim=True)
        
        # 4. 自適應軟性相位相關 (Soft Phase Correlation)
        # 若 magnitude 遠小於 mean_energy，該頻段乘子趨近於 0 (自動靜音雜訊)
        # 若 magnitude 遠大於 mean_energy，該頻段乘子趨近於長度為 1 的純相位
        dynamic_phase_multiplier = cross_power_spectrum / (magnitude + mean_energy + self.eps)
        
        # 5. 將動態相位套用到 V_f 上
        H_f = V_f * dynamic_phase_multiplier
        
        # 6. 轉回時域
        out = torch.fft.irfft(H_f, n=qk_tensor.size(1), dim=1)
        
        return out

# =========================================================
# 核心創新：Gated Dual-Pathway Fusion Block
# =========================================================
class GatedFusionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim, max_seq_len=512):
        super(GatedFusionBlock, self).__init__()
        
        # 替換為不對稱 Shared Attention 與零參數 Dynamic Hologram
        self.attention = AsymmetricSharedAttention(embed_dim, num_heads, max_seq_len)
        self.hologram = DynamicHologram()
        
        self.gate_linear = nn.Linear(embed_dim, 1)
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim)
        )
    
    def forward(self, x):
        gate = torch.sigmoid(self.gate_linear(x))
        
        # 關鍵突破：透過共用線性投射層，省下兩倍的 O(D^2) 參數量與運算
        shared_qk = self.attention.qk_linear(x)
        V = self.attention.v_linear(x)
        
        # 雙路徑並行 (傳遞已經算好的特徵)
        attn_out = self.attention.forward_with_precomputed(x, shared_qk, V)
        holo_out = self.hologram(shared_qk, V)
        
        fused = gate * attn_out + (1 - gate) * holo_out
        
        x = self.norm1(x + fused)
        
        ff_out = self.feed_forward(x)
        x = self.norm2(x + ff_out)
        
        return x

# =========================================================
# 最終模型：Phase-locked Holographic Transformer Fusion
# =========================================================
class PhaseLockedHolographicTransformerFusion(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, hidden_dim, num_blocks, max_seq_len=512):
        super(PhaseLockedHolographicTransformerFusion, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(max_seq_len, embed_dim)
        
        self.blocks = nn.ModuleList([
            GatedFusionBlock(embed_dim, num_heads, hidden_dim, max_seq_len)
            for _ in range(num_blocks)
        ])
        
        self.fc_out = nn.Linear(embed_dim, vocab_size)
    
    def forward(self, x):
        batch_size, seq_len = x.size()
        
        pos = torch.arange(0, seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len)
        out = self.embedding(x) + self.pos_embedding(pos)
        
        for block in self.blocks:
            out = block(out)
        
        logits = self.fc_out(out)
        return logits
    
    def get_gate_stats(self, x):
        batch_size, seq_len = x.size()
        pos = torch.arange(0, seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len)
        out = self.embedding(x) + self.pos_embedding(pos)
        
        gate_values = []
        for block in self.blocks:
            gate = torch.sigmoid(block.gate_linear(out))
            gate_values.append(gate.mean().item())
            out = block(out)
        
        return gate_values

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    dataset = CustomDataset(num_samples=5000, seq_len=SEQ_LEN)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = PhaseLockedHolographicTransformerFusion(
        vocab_size=VOCAB_SIZE, 
        embed_dim=EMBED_DIM, 
        num_heads=NUM_HEADS, 
        hidden_dim=HIDDEN_DIM, 
        num_blocks=2,
        max_seq_len=MAX_SEQ_LEN
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"PHTF (動態鎖相版) 總參數量: {total_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    print(f"\n開始訓練 PHTF 模型... 序列長度: {SEQ_LEN}")
    
    model.train()
    for epoch in range(NUM_EPOCHS):
        total_loss = 0
        correct = 0
        for batch_idx, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            logits = model(x)
            
            last_token_logits = logits[:, -1, :]
            
            loss = criterion(last_token_logits, y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            preds = last_token_logits.argmax(dim=-1)
            correct += (preds == y).sum().item()
            
        acc = correct / len(dataset)
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {total_loss / len(dataloader):.4f}, Acc: {acc:.4f}")

    # 分析 Gate 偏好
    print("\n--- Gate 分析 (Attention vs Hologram 偏好) ---")
    model.eval()
    with torch.no_grad():
        sample_x, sample_y = dataset[0]
        sample_x_batch = sample_x.unsqueeze(0).to(device)
        
        gate_stats = model.get_gate_stats(sample_x_batch)
        for i, g in enumerate(gate_stats):
            attn_pct = g * 100
            holo_pct = (1 - g) * 100
            print(f"  Block {i+1}: Attention {attn_pct:.1f}% | Hologram {holo_pct:.1f}%")
        
        logits = model(sample_x_batch)
        pred = logits[:, -1, :].argmax(dim=-1).item()
        
        print(f"\n輸入序列(前63個): {sample_x[:63].tolist()}")
        print(f"使用的 KEY: {sample_x[63].item()}")
        print(f"正確答案: {sample_y.item()}")
        print(f"PHTF 預測: {pred}")

if __name__ == "__main__":
    main()