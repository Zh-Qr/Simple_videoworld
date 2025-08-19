# -*- coding: utf-8 -*-
"""
Project: SnakeFrame Next‐Frame Prediction Training & Visualization
File Name: train_and_visualize_predictor.py
Created on: 2025/06/07
Author: fastbiubiu@163.com
Desc: 训练 Transformer 预测器 (token_t → token_{t+1})，并可视化下一帧预测结果
"""

import os
import time
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms as T
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.multiprocessing as mp

# ————————— VQ-VAE 定义 (只用到 encode/vq/dec) —————————
# ———————————————— 模型组件 ————————————————
class ConvBlock(nn.Module):
    """Conv→BN→ReLU"""
    def __init__(self, c_in, c_out, k=3, s=1, p=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, k, s, p, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)


class SnakeEncoder(nn.Module):
    """
    Downsample ×2 + Residual ×2 → 输出 [B, emb_dim, 16,16]
    """
    def __init__(self, emb_dim=64):
        super().__init__()
        self.enc = nn.Sequential(
            ConvBlock(3,  32, k=4, s=2, p=1),  # 64→32
            ConvBlock(32, 64, k=4, s=2, p=1),  # 32→16
            ConvBlock(64, 64),                 # 保持 16×16
            ConvBlock(64, 64),
        )
        self.project = nn.Conv2d(64, emb_dim, kernel_size=1)

    def forward(self, x):
        h = self.enc(x)
        return self.project(h)  # [B, emb_dim,16,16]


class SnakeDecoder(nn.Module):
    """
    对称 Decoder: [B,emb_dim,16,16] → [B,3,64,64]
    """
    def __init__(self, emb_dim=64):
        super().__init__()
        self.project = nn.Conv2d(emb_dim, 64, kernel_size=1)
        self.dec = nn.Sequential(
            ConvBlock(64, 64),
            ConvBlock(64, 32),
            nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1),  #16→32
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 3,  kernel_size=4, stride=2, padding=1),  #32→64
            nn.Sigmoid(),  # 输出 [0,1]
        )

    def forward(self, z_q):
        h = self.project(z_q)
        return self.dec(h)


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, beta: float = 0.25):
        super().__init__()
        self.K = num_embeddings
        self.D = embedding_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.K, self.D)
        self.embedding.weight.data.uniform_(-1/self.K, 1/self.K)

    def forward(self, z_e: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, D, H, W = z_e.shape
        flat = z_e.permute(0,2,3,1).reshape(-1, D)  # [B*H*W, D]

        # L2
        dist = (flat.pow(2).sum(1,keepdim=True)
                - 2 * flat @ self.embedding.weight.t()
                + self.embedding.weight.pow(2).sum(1))
        idx = dist.argmin(1)  # [B*H*W]
        z_q = self.embedding(idx).view(B, H, W, D).permute(0,3,1,2)

        # VQ loss
        loss_commit = F.mse_loss(z_q.detach(), z_e)
        loss_embed  = F.mse_loss(z_q, z_e.detach())
        vq_loss = loss_embed + self.beta * loss_commit

        # straight-through
        z_q = z_e + (z_q - z_e).detach()
        return z_q, vq_loss


class VQVAE(nn.Module):
    def __init__(self, num_embeddings=128, embedding_dim=64, beta=0.25):
        super().__init__()
        self.enc = SnakeEncoder(emb_dim=embedding_dim)
        self.vq  = VectorQuantizer(num_embeddings, embedding_dim, beta)
        self.dec = SnakeDecoder(emb_dim=embedding_dim)

    def encode(self, x: torch.Tensor) -> torch.LongTensor:
        z_e = self.enc(x)               # [B, D, H, W]
        B, D, H, W = z_e.shape
        flat = z_e.permute(0,2,3,1).reshape(-1, D)  # [B*H*W, D]
        # 注意这里是 self.vq.embedding
        dist = (flat.pow(2).sum(1, keepdim=True)
                - 2 * flat @ self.vq.embedding.weight.t()
                + self.vq.embedding.weight.pow(2).sum(1))
        idx = dist.argmin(1)                            # [B*H*W]
        return idx.view(B, H, W)                       # [B, H, W]

    def decode(self, idx: torch.LongTensor) -> torch.Tensor:
        B, H, W = idx.shape
        flat = idx.view(-1)
        # 这里也用 self.vq.embedding
        emb = self.vq.embedding.weight               # [K, D]
        z_q = emb[flat].view(B, H, W, -1).permute(0,3,1,2)  # [B, D, H, W]
        return self.dec(z_q)                         # [B, 3, 64, 64]

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z_e = self.enc(x)
        z_q, vq_loss = self.vq(z_e)
        x_hat = self.dec(z_q)
        return x_hat, vq_loss





# —————— 构造 (frame_t → frame_{t+1}) token 数据集 ——————
class SequenceFrameDataset(Dataset):
    def __init__(self, root_dir:str, vqvae:VQVAE, device, transform=None):
        self.vqvae = vqvae.to(device).eval()
        self.device = device
        self.transform = transform or T.Compose([
            T.Resize((64,64)), T.ToTensor()
        ])
        self.pairs = []
        for run_dir in sorted(Path(root_dir).glob('run_*')):
            frames = sorted(run_dir.glob('*.png'))
            for i in range(len(frames)-3):
                self.pairs.append((frames[i], frames[i+3]))
        print(f"Total frame‐pairs: {len(self.pairs)}")
    def __len__(self): return len(self.pairs)
    def __getitem__(self, idx):
        p0, p1 = self.pairs[idx]
        img0 = Image.open(p0).convert('RGB')
        img1 = Image.open(p1).convert('RGB')
        x0 = self.transform(img0).unsqueeze(0).to(self.device)
        x1 = self.transform(img1).unsqueeze(0).to(self.device)
        with torch.no_grad():
            idx0 = self.vqvae.encode(x0)  # [1,16,16]
            idx1 = self.vqvae.encode(x1)
        return idx0.view(-1), idx1.view(-1)  # ([256], [256])


# —————— Transformer 预测器 ——————
class TokenPredictor(nn.Module):
    def __init__(self, num_embeddings=256, emb_dim=64, n_head=8, num_layers=4):
        super().__init__()
        self.seq_len = 16*16
        self.token_emb = nn.Embedding(num_embeddings, emb_dim)
        self.pos_emb   = nn.Parameter(torch.randn(1, self.seq_len, emb_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=emb_dim, nhead=n_head,
            batch_first=True, dim_feedforward=emb_dim*4)
        self.transformer = nn.TransformerEncoder(layer, num_layers)
        self.head = nn.Linear(emb_dim, num_embeddings)
    def forward(self, seq:torch.LongTensor) -> torch.Tensor:
        # seq: [B, seq_len]
        x = self.token_emb(seq) + self.pos_emb  # [B,seq_len,D]
        x = self.transformer(x)                 # [B,seq_len,D]
        return self.head(x)                     # [B,seq_len,K]


# —————— 训练 Predictor ——————
def train_predictor(predictor, loader, device, epochs=10, lr=1e-4):
    predictor.to(device).train()
    optimizer = optim.Adam(predictor.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    for ep in range(1, epochs+1):
        total_loss = 0.0
        pbar = tqdm(loader, desc=f'Epoch {ep}/{epochs}', ncols=100)
        for seq0, seq1 in pbar:
            # seq0, seq1: [B,256]
            seq0 = seq0.to(device)
            seq1 = seq1.to(device)
            optimizer.zero_grad()
            logits = predictor(seq0)             # [B,256,K]
            loss = criterion(
                logits.view(-1, logits.size(-1)),  # [B*256, K]
                seq1.view(-1)                       # [B*256]
            )
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix(Loss=f'{total_loss/len(loader):.4f}')
        print(f'✓ Epoch {ep} avg_loss={total_loss/len(loader):.4f}')
    return predictor


# —————— 可视化 下一帧预测 ——————
def show_prediction(root_dir, vqvae, predictor, device, timestamp,
                    save_dir='fig_pred', max_n=4):
    os.makedirs(save_dir, exist_ok=True)
    vqvae.eval(); predictor.eval()

    # 取第一个 run，前 max_n 对帧
    run_dir = sorted(Path(root_dir).glob('run_*'))[0]
    frames = sorted(run_dir.glob('*.png'))
    selected = frames[: max_n+1]  # 需要 t and t+1

    transform = T.Compose([T.Resize((64,64)), T.ToTensor()])

    inputs, trues, preds = [], [], []
    with torch.no_grad():
        for i in range(max_n):
            img_t = Image.open(selected[i]).convert('RGB')
            img_tp1 = Image.open(selected[i+1]).convert('RGB')
            x_t = transform(img_t).unsqueeze(0).to(device)
            inputs.append(x_t.cpu().clamp(0,1).squeeze(0))

            # encode & predict next token
            idx_t = vqvae.encode(x_t).view(1, -1)                # [1,256]
            logits = predictor(idx_t)                            # [1,256,256]
            idx_next = logits.argmax(-1)                         # [1,256]
            idx_next2d = idx_next.view(1,16,16)

            # decode prediction
            x_pred = vqvae.decode(idx_next2d).cpu().clamp(0,1).squeeze(0)
            preds.append(x_pred)
            # true next
            x_true = transform(img_tp1).cpu().clamp(0,1)
            trues.append(x_true)

    # 绘制
    fig, axes = plt.subplots(max_n, 3,
                             figsize=(3*3, 3*max_n), squeeze=False)
    titles = ['Input t', 'Predicted t+1', 'True t+1']
    for i in range(max_n):
        axes[i][0].imshow(inputs[i].permute(1,2,0))
        axes[i][1].imshow(preds[i].permute(1,2,0))
        axes[i][2].imshow(trues[i].permute(1,2,0))
        for j in range(3):
            axes[i][j].axis('off')
            if i == 0:
                axes[i][j].set_title(titles[j])

    save_path = os.path.join(save_dir, f'prediction_{timestamp}.png')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f'✅ Next‐frame 可视化已保存: {save_path}')


# —————— 主流程 ——————
if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    timestamp = time.strftime('%Y%m%d_%H%M%S')

    # 1) 加载预训练 VQ-VAE
    vqvae = VQVAE(num_embeddings=128, embedding_dim=64).to(device)
    vqvae.load_state_dict(torch.load('weights/vqvae_20250607_155540.pth', map_location=device))

    # 2) 构建数据集 & loader
    ds = SequenceFrameDataset('../../../mnt/sdc/zhongqirui/gameruns', vqvae, device)
    loader = DataLoader(ds, batch_size=64, shuffle=True, num_workers=2)

    print("dataset is ready")

    # 3) 构建并训练 Transformer 预测器
    predictor = TokenPredictor(num_embeddings=256, emb_dim=64, n_head=8, num_layers=4)
    predictor = train_predictor(predictor, loader, device, epochs=20, lr=1e-4)

    # 4) 保存预测器权重
    os.makedirs('weights', exist_ok=True)
    torch.save(predictor.state_dict(), f'weights/predictor_{timestamp}.pth')
    print(f'💾 Predictor 权重已保存: weights/predictor_{timestamp}.pth')

    # 5) 可视化前 max_n 帧的预测效果
    show_prediction('../../../mnt/sdc/zhongqirui/gameruns', vqvae, predictor, device, timestamp,
                    save_dir='fig', max_n=16)
