import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# =================================================================
# PLHND — Phase-Locked Holography Network (Asymmetric Shared-QK + RoPE)
# 單通道全息版 (無 Multi-Head Attention)
# =================================================================
# 更新要點：
# 1. 廢除 Multi-Head 的特徵切分，讓頻域的互相關運算能一體化處理完整的特徵頻譜。
# 2. 保留 Asymmetric Shared-QK (qk_linear, v_linear, out_linear, k_scale) 與 RoPE。
# 3. 確保總參數量與 TFERE.py (330,624) 完全對齊。
# =================================================================

MAX_SEQ_LEN = 8192
SEQ_LEN = 64
BATCH_SIZE = 16
EMBED_DIM = 128
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
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cos(freqs)
    freqs_sin = torch.sin(freqs)
    return freqs_cos, freqs_sin

def apply_rotary_emb(xq, xk, freqs_cos, freqs_sin):
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
# 核心模組：Aligned Dynamic Phase-Locked Hologram (No Heads)
# =========================================================

class AlignedDynamicPhaseLockedHologram(nn.Module):
    """
    對齊版動態鎖相全息模組 (移除 Multi-Head)：
    1. 使用完整的 embed_dim 進行 Shared QK 投射。
    2. 將整個特徵視為一個完整的空間，套用 RoPE。
    3. 在頻域進行全局的互相關運算，保留最完整的頻率干涉。
    """
    def __init__(self, embed_dim, max_seq_len=512):
        super(AlignedDynamicPhaseLockedHologram, self).__init__()
        self.embed_dim = embed_dim
        self.eps = 1e-8
        
        # 對標 TFERE 的三大投射，保證參數數量一致
        self.qk_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.out_linear = nn.Linear(embed_dim, embed_dim)
        
        # O(D) 的不對稱擾動參數
        self.k_scale = nn.Parameter(torch.ones(embed_dim))

        # 註冊 RoPE buffer，直接以整個 embed_dim 進行旋轉
        freqs_cos, freqs_sin = precompute_freqs_cis(self.embed_dim, max_seq_len)
        self.register_buffer("freqs_cos", freqs_cos)
        self.register_buffer("freqs_sin", freqs_sin)

    def forward(self, x):
        B, seq_len, embed_dim = x.size()
        
        # 1. 產生 Shared QK 與 V (Content)
        shared_qk = self.qk_linear(x)
        V = self.v_linear(x)
        
        # 2. 透過 k_scale 產生不對稱的 Q 與 K
        Q = shared_qk
        K = shared_qk * self.k_scale
        
        # 3. 為了相容 apply_rotary_emb 內部的廣播機制，擴充一個虛擬的 head 維度
        Q_split = Q.unsqueeze(2)  # [B, seq_len, 1, embed_dim]
        K_split = K.unsqueeze(2)  # [B, seq_len, 1, embed_dim]
        
        # 4. 套用 RoPE
        f_cos = self.freqs_cos[:seq_len]
        f_sin = self.freqs_sin[:seq_len]
        Q_rot, K_rot = apply_rotary_emb(Q_split, K_split, f_cos, f_sin)
        
        # 擠掉虛擬的 head 維度，準備進入頻域
        Q_rot = Q_rot.squeeze(2)  # [B, seq_len, embed_dim]
        K_rot = K_rot.squeeze(2)  # [B, seq_len, embed_dim]
        
        # 5. 轉入頻域
        Q_f = torch.fft.rfft(Q_rot, dim=1)
        K_f = torch.fft.rfft(K_rot, dim=1)
        V_f = torch.fft.rfft(V, dim=1)
        
        # 6. 頻域互相關：計算互功率譜
        cross_power = Q_f * torch.conj(K_f)
        magnitude = torch.abs(cross_power)
        
        # 7. 自適應軟性相位相關 (加上 eps 避免奇異點)
        mean_energy = magnitude.mean(dim=1, keepdim=True)
        dynamic_phase = cross_power / (magnitude + mean_energy + self.eps)
        
        # 8. 將動態相位套用到 Content 上
        H_f = V_f * dynamic_phase
        
        # 9. 轉回時域
        out = torch.fft.irfft(H_f, n=seq_len, dim=1)
        
        # 10. 線性轉換整理特徵
        out = self.out_linear(out)
        
        return out

# =========================================================
# HologramBlock
# =========================================================

class HologramBlock(nn.Module):
    def __init__(self, embed_dim, hidden_dim, max_seq_len):
        super(HologramBlock, self).__init__()
        self.hologram = AlignedDynamicPhaseLockedHologram(embed_dim, max_seq_len)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim)
        )

    def forward(self, x):
        holo_out = self.hologram(x)
        x = self.norm1(x + holo_out)
        
        ff_out = self.feed_forward(x)
        x = self.norm2(x + ff_out)
        return x

# =========================================================
# 最終模型：PhaseLockedHolographyNetwork
# =========================================================

class PhaseLockedHolographyNetwork(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_blocks, max_seq_len=512):
        super(PhaseLockedHolographyNetwork, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(max_seq_len, embed_dim)
        
        self.blocks = nn.ModuleList([
            HologramBlock(embed_dim, hidden_dim, max_seq_len) 
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

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    dataset = CustomDataset(num_samples=5000, seq_len=SEQ_LEN)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 參數介面改回原本不需要 num_heads 的版本
    model = PhaseLockedHolographyNetwork(
        vocab_size=VOCAB_SIZE, 
        embed_dim=EMBED_DIM, 
        hidden_dim=HIDDEN_DIM, 
        num_blocks=2,
        max_seq_len=MAX_SEQ_LEN
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"PLHND (無 Heads + Shared-QK + RoPE) 總參數量: {total_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    print(f"\n開始訓練 PLHND 模型... 序列長度: {SEQ_LEN}")
    
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

    print("\n--- 測試 PLHND 預測 ---")
    model.eval()
    with torch.no_grad():
        sample_x, sample_y = dataset[0]
        sample_x_batch = sample_x.unsqueeze(0).to(device)
        
        logits = model(sample_x_batch)
        pred = logits[:, -1, :].argmax(dim=-1).item()
        
        print(f"輸入序列(前63個): {sample_x[:63].tolist()}")
        print(f"使用的 KEY: {sample_x[63].item()}")
        print(f"正確答案 (KEY 的下一個數字): {sample_y.item()}")
        print(f"PLHND 預測: {pred}")

if __name__ == "__main__":
    main()