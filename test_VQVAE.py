"""
Test the accuracy of the training result
CNN is introduced to test the 
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from PIL import Image
import numpy as np
import os
import torch.nn.functional as F
from typing import Tuple
import time
from tqdm import tqdm
import lpips
from torch.amp import GradScaler, autocast
import csv
import argparse



class AdditionDataset(Dataset):
    def __init__(self, img_npy, lab_npy, val_npy, transform=None):
        # 这里不再 cast to np.uint8！
        # 保存时 images.npy/labels.npy 已经是经过 Normalize((0.5,), (0.5,)) 的浮点张量
        self.images  = np.load(img_npy).astype(np.uint8)   # ensure uint8
        self.labels  = np.load(lab_npy).astype(np.uint8)
        self.values = np.load(val_npy).astype(np.int64)    # shape (N,), 真正的和
        self.t = transform
        assert len(self.images)==len(self.labels)==len(self.values)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # 直接从 numpy 变 torch.tensor
        img  = Image.fromarray(self.images[idx],  mode='L')
        lab  = Image.fromarray(self.labels[idx], mode='L')
        val = int(self.values[idx])
        if self.t:
            img = self.t(img)
            lab = self.t(lab)
        return img, lab, val
    
class StaticAdditionDataset(Dataset):
    def __init__(self, img_npy, lab_npy, val_npy, transformer):
        # 直接读回 float32（-1~1）
        self.images = np.load(img_npy).astype(np.float32)    # shape (N,H,W)
        self.images8 = np.load(img_npy).astype(np.uint8)    # shape (N,H,W)
        self.labels = np.load(lab_npy).astype(np.float32)
        self.values = np.load(val_npy).astype(np.int64)
        self.t = transformer
        assert len(self.images)==len(self.labels)==len(self.values)

    def __len__(self):
        return len(self.values)

    def __getitem__(self, idx):
        img = torch.from_numpy(self.images[idx]).unsqueeze(0)  # [1,H,W]，已在 [-1,1]
        img8 = Image.fromarray(self.images8[idx],  mode='L')
        if self.t:
            img8 = self.t(img8)
        lab = torch.from_numpy(self.labels[idx]).unsqueeze(0)
        val = int(self.values[idx])
        return img, img8, lab, val
    
tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.pool  = nn.MaxPool2d(2, 2)
        self.fc1   = nn.Linear(64 * 7 * 7, 128)
        self.fc2   = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # 28→14
        x = self.pool(F.relu(self.conv2(x)))   # 14→7
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int,
                 commitment_cost: float = 0.25):
        super().__init__()
        self.D = embedding_dim
        self.K = num_embeddings
        self.beta = commitment_cost

        self.embeddings = nn.Embedding(self.K, self.D)
        self.embeddings.weight.data.uniform_(-1 / self.K, 1 / self.K)

    def forward(self, z_e: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # z_e: [B, D, H, W]  (这里 H=7, W=14)
        B, D, H, W = z_e.shape
        z_e_flat = z_e.permute(0, 2, 3, 1).reshape(-1, D)  # [BHW, D]

        # L2 距离
        dist = (z_e_flat.pow(2).sum(1, keepdim=True)
               -2 * z_e_flat @ self.embeddings.weight.t()
               + self.embeddings.weight.pow(2).sum(1))

        idx = dist.argmin(1)                               # 最近邻索引
        z_q = self.embeddings(idx).view(B, H, W, D).permute(0, 3, 1, 2)

        # VQ‑Loss
        loss_codebook  = F.mse_loss(z_q.detach(), z_e)
        loss_commit    = F.mse_loss(z_q, z_e.detach())
        vq_loss = loss_commit + self.beta * loss_codebook

        # Straight‑Through
        z_q = z_e + (z_q - z_e).detach()
        return z_q, vq_loss
    
class TokenTransformerLite(nn.Module):
    """
    输入  [B,D,7,14]  → 98 token
    输出  [B,D,7,14]（同分辨率）
    仅做轻量增强：层数 6、Pre‑Norm、FeedForward×6、Dropout0.1
    """
    def __init__(self,
                 emb_dim:      int = 64,
                 num_layers:   int = 6,     # ← 4→6
                 n_head:       int = 8,
                 use_sin_pe:   bool = False):
        super().__init__()
        self.h, self.w  = 7, 14
        self.seq_len    = self.h * self.w     # 98

        if use_sin_pe:
            # 固定正弦 PE：不训练、无额外参数
            pe = self.build_sinusoid_pe(self.seq_len, emb_dim)
            self.register_buffer('pe', pe, persistent=False)
            self.learnable_pos = None
        else:
            self.learnable_pos = nn.Parameter(torch.randn(1, self.seq_len, emb_dim))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=n_head,
            batch_first=True,
            dim_feedforward=emb_dim * 6,     # 4× → 6×
            dropout=0.1,
            norm_first=True                  # Pre‑Norm
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers)
        self.norm = nn.LayerNorm(emb_dim)

    @staticmethod
    def build_sinusoid_pe(seq_len, dim):
        pos = torch.arange(seq_len)[:, None]
        i   = torch.arange(dim)[None, :]
        angle = pos / torch.pow(10000, (2*(i//2)) / dim)
        pe   = torch.zeros(seq_len, dim)
        pe[:, 0::2] = torch.sin(angle[:, 0::2])
        pe[:, 1::2] = torch.cos(angle[:, 1::2])
        return pe.unsqueeze(0)   # [1,seq,dim]

    def forward(self, z_q):                     # z_q [B,D,7,14]
        B, D, H, W = z_q.shape
        x = z_q.flatten(2).permute(0, 2, 1)     # [B,98,D]

        if self.learnable_pos is not None:
            x = x + self.learnable_pos
        else:
            x = x + self.pe.to(x.dtype)

        x = self.encoder(x)
        x = self.norm(x)                        # [B,98,D]
        x = x.permute(0, 2, 1).view(B, D, H, W)
        return x
    
class Encoder(nn.Module):
    def __init__(self, in_ch=1, hid=128, emb_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch,  hid,     kernel_size=4, stride=2, padding=1),   # 28×56 → 14×28
            nn.ReLU(),
            nn.Conv2d(hid,    hid*2,   kernel_size=4, stride=2, padding=1),   # 14×28 → 7×14
            nn.ReLU(),
            nn.Conv2d(hid*2,  hid*2,   kernel_size=3, stride=1, padding=1),   # 尺寸不变
            nn.ReLU(),
            nn.Conv2d(hid*2,  emb_dim, kernel_size=1)                         # 1×1，得到 D
        )

    def forward(self, x):
        return self.net(x)      # [B, emb_dim, 7, 14]
    
class Decoder(nn.Module):
    def __init__(self, emb_dim=64, hid=128, out_ch=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(emb_dim, hid*2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(hid*2, hid,  kernel_size=4, stride=2, padding=1),          # 7×14 → 14×28
            nn.ReLU(),
            # 仅把高度翻倍：stride=(2,1)、kernel=(4,3)
            nn.ConvTranspose2d(hid, out_ch, kernel_size=(4,3), stride=(2,1), padding=(1,1)),
            nn.Tanh()   # 若前端做了 [-1,1] 归一化；如用 [0,1] 则改 Sigmoid
        )

    def forward(self, z_q):
        return self.net(z_q)    # [B,1,28,28]
    
class VQVAE(nn.Module):
    def __init__(self,
                 num_embeddings=256,
                 embedding_dim=64,
                 commitment_cost=0.25,
                 hid=128,
                 tf_layers=4):
        super().__init__()
        self.enc = Encoder(1, hid, embedding_dim)
        self.vq  = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        self.tf = TokenTransformerLite(emb_dim=64,
                               num_layers=6,   # 默认=6
                               n_head=8,
                               use_sin_pe=False)
        self.dec = Decoder(embedding_dim, hid, 1)

    def forward(self, x):
        z_e = self.enc(x)               # [B, D, 7,14]
        z_q, vq_loss = self.vq(z_e)
        z_tf = self.tf(z_q)
        x_hat = self.dec(z_tf)           # [B,1,28,28]
        return x_hat, vq_loss
    

@torch.no_grad()
def show_batch_effect(loader, vqvae, cnn, device, save_path, transform, max_vis=8):
    vqvae.eval(); cnn.eval()

    # loader 返回的是已经经过 transform_vq 的 imgs, labs ∈ [-1,1]
    # imgs, imgs8, labs, vals = next(iter(loader))
    # imgs, imgs8, labs = imgs.to(device), imgs8.to(device), labs.to(device)

    imgs, labs, vals = next(iter(loader))
    imgs, labs = imgs.to(device), labs.to(device)


    recon, _  = vqvae(imgs)
    recon_inv = -recon

    # (2) 反归一化到 [0,1] → 用于展示
    imgs_disp  = (imgs  * 0.5 + 0.5).clamp(0,1).cpu()
    recon_disp = (recon_inv* 0.5 + 0.5).clamp(0,1).cpu()
    labs_disp  = (labs  * 0.5 + 0.5).clamp(0,1).cpu()


    # (4) CNN 预测
    pred_recon = cnn(recon_inv).argmax(1).cpu().numpy()
    pred_lab   = cnn(-labs).argmax(1).cpu().numpy()

    # (5) 可视化并保存
    B = min(max_vis, imgs.size(0))
    plt.figure(figsize=(9, 3*B))
    for i in range(B):
        plt.subplot(B,3,3*i+1)
        plt.imshow(imgs_disp [i,0], cmap='gray'); plt.axis('off'); plt.title("Input")

        plt.subplot(B,3,3*i+2)
        plt.imshow(recon_disp[i,0], cmap='gray'); plt.axis('off')
        plt.title(f"Recon\n(p={pred_recon[i]}, gt={vals[i]})")

        plt.subplot(B,3,3*i+3)
        plt.imshow(labs_disp [i,0], cmap='gray'); plt.axis('off')
        plt.title(f"Label\n(p={pred_lab[i]}, gt={vals[i]})")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def calculate_reconstruction_consistency(vqvae, cnn, loader, device, transform):
    """
    遍历 loader：
      1) imgs, labs ∈ [-1,1] 直接送 VQ-VAE 得到 recon ∈ [-1,1]
      2) recon 和 labs 反归一化到 [0,1] 再 Normalize((0.5,), (0.5,)) → [-1,1] 送 CNN
      3) 统计 CNN(recon) vs. CNN(labs) 与真实 sum 值 vals 的一致率
    """
    vqvae.eval(); cnn.eval()
    total = match = 0
    for imgs, labs, vals in tqdm(loader, desc="Consistency"):
        imgs, labs = imgs.to(device), labs.to(device)
        recon, _  = vqvae(imgs)

        pr = cnn(-recon).argmax(1).cpu().numpy()
        pl = cnn(labs).argmax(1).cpu().numpy()
        vals_tensor = torch.tensor(vals, dtype=torch.long)

        match += (pr == vals_tensor).sum().item()
        total += imgs.size(0)

    rate = match / total if total else 0
    print(f"一致率: {match}/{total} = {rate*100:.2f}%")
    return rate


if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    # Initialize the model, loss function and optimizer
    state_dict_VQVAE = torch.load('weight/vqvae_best_2025_05_04_13_52_47.pth')
    model = VQVAE(num_embeddings=64, embedding_dim=64, hid=128).to(device)
    model.load_state_dict(state_dict_VQVAE)

    # ------- 改进写法 -------
    state_dict_CNN = torch.load('weight/MNIST_best.pth',
                            map_location='cpu',
                            weights_only=True)
    verify_model = SimpleCNN().to(device)
    verify_model.load_state_dict(state_dict_CNN)
        
    data_set = AdditionDataset(img_npy='../../../../mnt/sdc/zhongqirui/static_MNIST/images.npy',
            lab_npy='../../../../mnt/sdc/zhongqirui/static_MNIST/labels.npy',
            val_npy='../../../../mnt/sdc/zhongqirui/static_MNIST/values.npy',
            transform=tf)

    # data_set = StaticAdditionDataset(
    #     img_npy='../../../../mnt/sdc/zhongqirui/static_MNIST/images.npy',
    #     lab_npy='../../../../mnt/sdc/zhongqirui/static_MNIST/labels.npy',
    #     val_npy='../../../../mnt/sdc/zhongqirui/static_MNIST/values.npy',
    #     transformer=tf
    # )

    data_loader = DataLoader(data_set, batch_size=16, shuffle=True)

    # ---------- 调用 ----------
    # 举例，生成 5 组可视化
    for idx in range(5):
        out_file = f'test_fig/test_{idx}.png'
        show_batch_effect(
            loader=data_loader,
            vqvae=model,
            cnn=verify_model,
            device=device,
            save_path=out_file,
            transform=tf,
            max_vis=16
        )
        print(f'Saved {out_file}')

    calculate_reconstruction_consistency(model, verify_model, data_loader, device,tf)