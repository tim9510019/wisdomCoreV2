import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# =================================================================
# Hybrid_Holo_Tfer — 異質混合架構：Hologram(L1) + Transformer(L2)
# =================================================================
# 實驗目標：
# 1. 驗證「先頻域鎖相 (Holo) 再時域注意 (Tfer)」的學習效率。
# 2. 保持與先前模型一致的參數對齊 (Shared-QK, RoPE, k_scale)。
# =================================================================

MAX_SEQ_LEN = 8192
SEQ_LEN = 64
BATCH_SIZE = 16
EMBED_DIM = 128
NUM_HEADS = 4
HIDDEN_DIM = 256
VOCAB_SIZE = 128
NUM_EPOCHS = 20


# =========================================================
# 數據集與輔助函數 (保持不變以利公平比較)
# =========================================================
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
            self.data_x[i, : self.seq_len - 1] = seq
            self.data_x[i, self.seq_len - 1] = key
            self.data_y[i] = target

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data_x[idx], self.data_y[idx]


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    return torch.cos(freqs), torch.sin(freqs)


def apply_rotary_emb(xq, xk, freqs_cos, freqs_sin):
    xq_r, xq_i = xq.float().reshape(xq.shape[:-1] + (-1, 2)).unbind(-1)
    xk_r, xk_i = xk.float().reshape(xk.shape[:-1] + (-1, 2)).unbind(-1)
    f_cos = freqs_cos.view(1, xq.shape[1], 1, -1)
    f_sin = freqs_sin.view(1, xq.shape[1], 1, -1)
    xq_out = torch.stack(
        [xq_r * f_cos - xq_i * f_sin, xq_r * f_sin + xq_i * f_cos], dim=-1
    ).flatten(3)
    xk_out = torch.stack(
        [xk_r * f_cos - xk_i * f_sin, xk_r * f_sin + xk_i * f_cos], dim=-1
    ).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


# =========================================================
# 第一層組件：Aligned Dynamic Hologram
# =========================================================
class AlignedHologramLayer(nn.Module):
    def __init__(self, embed_dim, max_seq_len):
        super().__init__()
        self.qk_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.out_linear = nn.Linear(embed_dim, embed_dim)
        self.k_scale = nn.Parameter(torch.ones(embed_dim))
        self.eps = 1e-8
        f_cos, f_sin = precompute_freqs_cis(embed_dim, max_seq_len)
        self.register_buffer("freqs_cos", f_cos)
        self.register_buffer("freqs_sin", f_sin)

    def forward(self, x):
        B, L, D = x.size()
        sqk = self.qk_linear(x)
        V = self.v_linear(x)
        Q_rot, K_rot = apply_rotary_emb(
            sqk.unsqueeze(2),
            (sqk * self.k_scale).unsqueeze(2),
            self.freqs_cos[:L],
            self.freqs_sin[:L],
        )

        Q_f, K_f, V_f = (
            torch.fft.rfft(Q_rot.squeeze(2), dim=1),
            torch.fft.rfft(K_rot.squeeze(2), dim=1),
            torch.fft.rfft(V, dim=1),
        )
        cp = Q_f * torch.conj(K_f)
        mag = torch.abs(cp)
        dp = cp / (mag + mag.mean(dim=1, keepdim=True) + self.eps)
        out = torch.fft.irfft(V_f * dp, n=L, dim=1)
        return self.out_linear(out)


# =========================================================
# 第二層組件：Aligned Attention
# =========================================================
class AlignedAttentionLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, max_seq_len):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.qk_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.out_linear = nn.Linear(embed_dim, embed_dim)
        self.k_scale = nn.Parameter(torch.ones(embed_dim))
        f_cos, f_sin = precompute_freqs_cis(self.head_dim, max_seq_len)
        self.register_buffer("freqs_cos", f_cos)
        self.register_buffer("freqs_sin", f_sin)

    def forward(self, x):
        B, L, D = x.size()
        sqk = self.qk_linear(x)
        V = self.v_linear(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        Q_rot, K_rot = apply_rotary_emb(
            sqk.view(B, L, self.num_heads, self.head_dim),
            (sqk * self.k_scale).view(B, L, self.num_heads, self.head_dim),
            self.freqs_cos[:L],
            self.freqs_sin[:L],
        )
        scores = torch.matmul(
            Q_rot.transpose(1, 2), K_rot.transpose(1, 2).transpose(-2, -1)
        ) / (self.head_dim**0.5)
        attn = torch.matmul(torch.softmax(scores, dim=-1), V)
        return self.out_linear(attn.transpose(1, 2).reshape(B, L, D))


# =========================================================
# 混合模型：HoloTfer
# =========================================================
class HoloTferHybrid(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, hidden_dim, max_seq_len=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(max_seq_len, embed_dim)

        # Block 1: Hologram
        self.h_layer = AlignedHologramLayer(embed_dim, max_seq_len)
        self.h_norm1 = nn.LayerNorm(embed_dim)
        self.h_ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.h_norm2 = nn.LayerNorm(embed_dim)

        # Block 2: Transformer
        self.t_layer = AlignedAttentionLayer(embed_dim, num_heads, max_seq_len)
        self.t_norm1 = nn.LayerNorm(embed_dim)
        self.t_ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.t_norm2 = nn.LayerNorm(embed_dim)

        self.fc_out = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        L = x.size(1)
        x = self.embedding(x) + self.pos_emb(torch.arange(L, device=x.device))

        # L1: Holo
        x = self.h_norm1(x + self.h_layer(x))
        x = self.h_norm2(x + self.h_ffn(x))

        # L2: Tfer
        x = self.t_norm1(x + self.t_layer(x))
        x = self.t_norm2(x + self.t_ffn(x))

        return self.fc_out(x)


# =========================================================
# 執行訓練
# =========================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = CustomDataset(5000, SEQ_LEN)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = HoloTferHybrid(VOCAB_SIZE, EMBED_DIM, NUM_HEADS, HIDDEN_DIM).to(device)
    print(
        f"Hybrid HoloTfer 總參數量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss, correct = 0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)[:, -1, :]
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct += (logits.argmax(1) == y).sum().item()
        print(
            f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {total_loss/len(loader):.4f}, Acc: {correct/5000:.4f}"
        )


if __name__ == "__main__":
    main()
