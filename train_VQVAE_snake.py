# -*- coding: utf-8 -*-
"""
Project: SnakeFrame VQ-VAE
File Name: train_snake_vqvae.py
Created on: 2025/06/07
Author: fastbiubiu@163.com
Desc: 用贪吃蛇帧数据训练 VQ-VAE，包含训练、保存权重和 batch 可视化
"""

import os
import glob
import time
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

# ———————————————— 数据集 ————————————————
class SnakeFrameDataset(Dataset):
    """
    递归读取 runs/run_*/*.png 中的所有游戏帧
    """
    def __init__(self, root_dir: str, transform=None):
        super().__init__()
        self.paths = sorted(glob.glob(os.path.join(root_dir, 'run_*', '*.png')))
        self.transform = transform or T.Compose([
            T.Resize((64, 64)),   # 调整到 64×64
            T.ToTensor(),         # 归一化到 [0,1]
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        x = self.transform(img)  # [3,64,64]
        return x, 0               # label 不用，只占位


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


class SnakeVQVAE(nn.Module):
    def __init__(self, num_embeddings=128, embedding_dim=64, beta=0.25):
        super().__init__()
        self.enc = SnakeEncoder(emb_dim=embedding_dim)
        self.vq  = VectorQuantizer(num_embeddings, embedding_dim, beta)
        self.dec = SnakeDecoder(embedding_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z_e = self.enc(x)               # [B,emb_dim,16,16]
        z_q, vq_loss = self.vq(z_e)     # [B,emb_dim,16,16], scalar
        x_hat = self.dec(z_q)           # [B,3,64,64]
        return x_hat, vq_loss


# ———————————————— 训练与可视化 ————————————————
def train_vqvae(model: nn.Module,
                loader: DataLoader,
                epochs: int,
                lr: float,
                device):
    model.to(device).train()
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.99))

    for ep in range(1, epochs+1):
        tot = rec_sum = vq_sum = 0.0
        pbar = tqdm(loader, desc=f'Epoch {ep}/{epochs}', ncols=100)
        for i, (imgs, _) in enumerate(pbar, 1):
            imgs = imgs.to(device)
            optimizer.zero_grad()           # 清零梯度
            recons, vq_loss = model(imgs)
            rec_loss = F.mse_loss(recons, imgs)
            loss = rec_loss + vq_loss
            loss.backward()
            optimizer.step()                # 更新参数

            tot     += loss.item()
            rec_sum += rec_loss.item()
            vq_sum  += vq_loss.item()
            pbar.set_postfix(
                Loss = f'{tot/i:.4f}',
                Recon= f'{rec_sum/i:.4f}',
                VQ   = f'{vq_sum/i:.4f}'
            )
        print(f'✓ Epoch {ep}  AvgLoss={tot/len(loader):.4f} '
              f'Recon={rec_sum/len(loader):.4f}  VQ={vq_sum/len(loader):.4f}')
    print('✅ Training completed!')



def show_batch(loader: DataLoader,
               model: nn.Module,
               device,
               timestamp: str,
               save_dir: str = 'fig',
               max_n: int = 8):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    with torch.no_grad():
        imgs, _ = next(iter(loader))
        imgs = imgs.to(device)
        recons, _ = model(imgs)

    # 反归一化
    imgs   = imgs.cpu().clamp(0,1)
    recons = recons.cpu().clamp(0,1)

    n = min(max_n, imgs.size(0))
    fig, axes = plt.subplots(n, 2, figsize=(4,2*n), squeeze=False)
    for i in range(n):
        axes[i][0].imshow(imgs[i].permute(1,2,0))
        axes[i][0].axis('off')
        axes[i][1].imshow(recons[i].permute(1,2,0))
        axes[i][1].axis('off')
        if i==0:
            axes[i][0].set_title('Input')
            axes[i][1].set_title('Recons')
    save_path = os.path.join(save_dir, f'result_{timestamp}.png')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f'✅ batch 可视化已保存: {save_path}')


if __name__ == '__main__':
    # 参数配置
    root_dir     = '../../../mnt/sdc/zhongqirui/gameruns'          # 贪吃蛇帧根目录
    batch_size   = 64
    epochs       = 10
    lr           = 2e-4
    num_embed    = 128
    emb_dim      = 64
    beta         = 0.25

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    timestamp = time.strftime('%Y%m%d_%H%M%S')

    # 数据
    dataset = SnakeFrameDataset(root_dir=root_dir,
                                transform=T.Compose([
                                    T.Resize((64,64)),
                                    T.ToTensor(),
                                ]))
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    print(f"Data set is ready, total frames: {len(dataset)}")

    # 模型
    model = SnakeVQVAE(num_embeddings=num_embed,
                       embedding_dim=emb_dim,
                       beta=beta).to(device)

    # 训练
    print("🚀 Starting training...")
    train_vqvae(model, loader, epochs, lr, device)

    # 保存权重
    os.makedirs('weights', exist_ok=True)
    weight_path = f'weights/vqvae_{timestamp}.pth'
    torch.save(model.state_dict(), weight_path)
    print(f'💾 权重已保存: {weight_path}')

    # 可视化一个 batch
    show_batch(loader, model, device, timestamp, save_dir='fig', max_n=8)
