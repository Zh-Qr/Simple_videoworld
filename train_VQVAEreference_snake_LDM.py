# -*- coding: utf-8 -*-
"""
File: train_and_visualize_predictor_with_ldm.py
Created on: 2025/06/17
Author: fastbiubiu@163.com
Desc: 在原有 VQ-VAE + Transformer 预测器 基础上，集成 VideoWorld LDM 模块  
      1) 训练 LDM（多步潜在动力学模型）  
      2) 冻结 VQ-VAE & LDM，用 LDM 输出的潜在码增强 Transformer 输入  
      3) 可视化下一帧预测效果  
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

# —————————————— VQ-VAE 定义 ——————————————
class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out, k=3, s=1, p=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, k, s, p, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)

class SnakeEncoder(nn.Module):
    def __init__(self, emb_dim=64):
        super().__init__()
        self.enc = nn.Sequential(
            ConvBlock(3, 32, k=4, s=2, p=1),
            ConvBlock(32, 64, k=4, s=2, p=1),
            ConvBlock(64, 64),
            ConvBlock(64, 64),
        )
        self.project = nn.Conv2d(64, emb_dim, 1)
    def forward(self, x):
        h = self.enc(x)
        return self.project(h)  # [B,emb_dim,16,16]

class SnakeDecoder(nn.Module):
    def __init__(self, emb_dim=64):
        super().__init__()
        self.project = nn.Conv2d(emb_dim, 64, 1)
        self.dec = nn.Sequential(
            ConvBlock(64,64),
            ConvBlock(64,32),
            nn.ConvTranspose2d(32,32,4,2,1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32,3,4,2,1),
            nn.Sigmoid(),
        )
    def forward(self, z_q):
        return self.dec(self.project(z_q))

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings:int, embedding_dim:int, beta:float=0.25):
        super().__init__()
        self.K, self.D, self.beta = num_embeddings, embedding_dim, beta
        self.embedding = nn.Embedding(self.K, self.D)
        nn.init.uniform_(self.embedding.weight, -1/self.K, 1/self.K)
    def forward(self, z_e:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B,D,H,W = z_e.shape
        flat = z_e.permute(0,2,3,1).reshape(-1, D)
        dist = (flat.pow(2).sum(1,True)
                - 2*flat@self.embedding.weight.t()
                + self.embedding.weight.pow(2).sum(1))
        idx = dist.argmin(1)
        z_q = self.embedding(idx).view(B,H,W,D).permute(0,3,1,2)
        loss_commit = F.mse_loss(z_q.detach(), z_e)
        loss_embed  = F.mse_loss(z_q, z_e.detach())
        vq_loss = loss_embed + self.beta * loss_commit
        z_q = z_e + (z_q - z_e).detach()
        return z_q, vq_loss

class VQVAE(nn.Module):
    def __init__(self, num_embeddings=128, embedding_dim=64, beta=0.25):
        super().__init__()
        self.enc = SnakeEncoder(embedding_dim)
        self.vq  = VectorQuantizer(num_embeddings, embedding_dim, beta)
        self.dec = SnakeDecoder(embedding_dim)
    def encode(self, x:torch.Tensor) -> torch.LongTensor:
        z_e = self.enc(x)
        B,D,H,W = z_e.shape
        flat = z_e.permute(0,2,3,1).reshape(-1, D)
        dist = (flat.pow(2).sum(1,True)
                -2*flat@self.vq.embedding.weight.t()
                +self.vq.embedding.weight.pow(2).sum(1))
        idx = dist.argmin(1)
        return idx.view(B,H,W)
    def decode(self, idx:torch.LongTensor) -> torch.Tensor:
        B,H,W = idx.shape
        flat = idx.view(-1)
        emb = self.vq.embedding.weight
        z_q = emb[flat].view(B,H,W,-1).permute(0,3,1,2)
        return self.dec(z_q)
    def forward(self, x:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z_e = self.enc(x)
        z_q, vq_loss = self.vq(z_e)
        return self.dec(z_q), vq_loss

# —————————————— LDM 定义 ——————————————
class CausalEncoder3D(nn.Module):
    """简易因果3D-CNN: [B,3,H+1,64,64]→[B,Feat,H+1,16,16]"""
    def __init__(self, feat_dim=64, seq_len=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(3,32,(3,4,4),(1,2,2),(1,1,1)),
            nn.BatchNorm3d(32), nn.ReLU(True),
            nn.Conv3d(32,64,(3,4,4),(1,2,2),(1,1,1)),
            nn.BatchNorm3d(64), nn.ReLU(True),
        )
        self.proj = nn.Conv3d(64, feat_dim, 1)
    def forward(self, x):
        return self.proj(self.net(x))

class LatentDynamicsModel(nn.Module):
    """LDM: 多步潜在动力学模型"""
    def __init__(self, feat_dim=64, H=4, num_embeddings=128, beta=0.25):
        super().__init__()
        self.H = H
        self.encoder3d = CausalEncoder3D(feat_dim, seq_len=H+1)
        self.queries = nn.Parameter(torch.randn(H, feat_dim,1,1))
        self.embed   = nn.Conv2d(feat_dim, feat_dim, 1)
        self.beta    = beta
        self.codebook = nn.Embedding(num_embeddings, feat_dim)
        nn.init.uniform_(self.codebook.weight, -1/num_embeddings, 1/num_embeddings)
    def forward(self, x_seq:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x_seq: [B,3,H+1,64,64]
        return z_q [B,H,feat,16,16], vq_loss
        """
        B = x_seq.size(0)
        feat3d = self.encoder3d(x_seq)      # [B,feat,H+1,16,16]
        f_t = feat3d[:,:, -1]               # [B,feat,16,16]
        losses, z_q_list = [], []
        for h in range(self.H):
            q = self.queries[h]             # [feat,1,1]
            z_tilde = f_t * q               # 跨时“注意力”
            z_e = self.embed(z_tilde)
            flat = z_e.permute(0,2,3,1).reshape(-1, z_e.size(1))
            dist = (flat.pow(2).sum(1,True)
                  -2*flat@self.codebook.weight.t()
                  +self.codebook.weight.pow(2).sum(1))
            idx = dist.argmin(1)
            z_q = self.codebook(idx).view(B,16,16,-1).permute(0,3,1,2)
            loss_c = F.mse_loss(z_q.detach(), z_e)
            loss_e = F.mse_loss(z_q, z_e.detach())
            losses.append(loss_e + self.beta * loss_c)
            z_q = z_e + (z_q - z_e).detach()
            z_q_list.append(z_q)
        vq_loss = torch.stack(losses).mean()
        z_q = torch.stack(z_q_list, dim=1)  # [B,H,feat,16,16]
        return z_q, vq_loss

# —————— 构造 LDM 训练集 ——————
class SequenceRawDataset(Dataset):
    """用于 LDM：读取 H+1 帧原始图像序列"""
    def __init__(self, root_dir:str, seq_len:int=5, transform=None):
        self.seq_len = seq_len
        self.transform = transform or T.Compose([T.Resize((64,64)), T.ToTensor()])
        self.windows = []
        for run in sorted(Path(root_dir).glob('run_*')):
            frames = sorted(run.glob('*.png'))
            for i in range(len(frames)-seq_len+1):
                self.windows.append(frames[i:i+seq_len])
        print(f"Raw sequences: {len(self.windows)}")
    def __len__(self): return len(self.windows)
    def __getitem__(self, idx):
        paths = self.windows[idx]
        imgs = [self.transform(Image.open(p).convert('RGB')).unsqueeze(1) for p in paths]
        x_seq = torch.cat(imgs, dim=1)  # [3, seq_len,64,64]
        return x_seq  # 返回 [3, H+1, 64,64]

# —————— 构造 Predictor 训练集 ——————
class SequenceSeqDataset(Dataset):
    """
    用于预测器：同时提供初始帧 token、目标帧 token 及原始序列
    """
    def __init__(self, root_dir:str, vqvae:VQVAE, ldm:LatentDynamicsModel,
                 device, seq_len:int=5, transform=None):
        self.vqvae = vqvae.to(device).eval()
        self.ldm   = ldm.to(device).eval()
        self.device = device
        self.seq_len = seq_len
        self.transform = transform or T.Compose([T.Resize((64,64)), T.ToTensor()])
        self.windows = []
        for run in sorted(Path(root_dir).glob('run_*')):
            frames = sorted(run.glob('*.png'))
            for i in range(len(frames)-seq_len+1):
                self.windows.append(frames[i:i+seq_len])
        print(f"Predictor samples: {len(self.windows)}")
    def __len__(self): return len(self.windows)
    def __getitem__(self, idx):
        paths = self.windows[idx]  # e.g. 5 帧路径
        # 1) 讀原始圖像並 transform
        frames = [ self.transform(Image.open(p).convert('RGB')).to(self.device)
                   for p in paths ]     # list of [3,64,64]
        # 2) 堆疊成 [seq_len, 3, 64, 64]
        frames = torch.stack(frames, dim=0)  

        # 3) 取首尾帧，添加 batch 维度，送入 VQ-VAE
        x0 = frames[0].unsqueeze(0)         # [1,3,64,64]
        xT = frames[-1].unsqueeze(0)        # [1,3,64,64]
        with torch.no_grad():
            idx0 = self.vqvae.encode(x0).view(-1)  # [256]
            idx1 = self.vqvae.encode(xT).view(-1)

        # 4) 准备给 LDM：需要 [B=1, C=3, D=seq_len, H=64, W=64]
        x_seq = frames.permute(1,0,2,3).unsqueeze(0)  
        #     from [seq_len,3,64,64] to [1,3,seq_len,64,64]
        with torch.no_grad():
            z_q, _ = self.ldm(x_seq)       # z_q: [1,H,feat,16,16]

        # 5) 拉平成 [H, feat*16*16] 作为 predictor 输入
        H = z_q.size(1)
        z_flat = z_q.view(1, H, -1).squeeze(0)  # [H, feat*16*16]

        return idx0.long(), idx1.long(), z_flat.float()

# —————— 扩展的 Transformer 预测器 ——————
class TokenPredictorWithLDM(nn.Module):
    def __init__(self, num_embeddings=256, emb_dim=64, n_head=8, num_layers=4, H=4, feat_dim=64):
        super().__init__()
        self.base_len = 16*16
        self.H = H
        self.token_emb = nn.Embedding(num_embeddings, emb_dim)
        self.ldm_proj  = nn.Linear(feat_dim*16*16, emb_dim)
        total_len = self.base_len + H
        self.pos_emb   = nn.Parameter(torch.randn(1, total_len, emb_dim))
        layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=n_head,
                                           batch_first=True,
                                           dim_feedforward=emb_dim*4)
        self.transformer = nn.TransformerEncoder(layer, num_layers)
        self.head = nn.Linear(emb_dim, num_embeddings)

    def forward(self, idx:torch.LongTensor, z_flat:torch.Tensor) -> torch.Tensor:
        """
        idx: [B,256], z_flat: [B,H, feat*16*16]
        """
        tok_emb = self.token_emb(idx)             # [B,256,emb_dim]
        z_emb = self.ldm_proj(z_flat)             # [B,H,emb_dim]
        # 作为额外“时间步”拼接
        x = torch.cat([tok_emb, z_emb], dim=1)    # [B,256+H,emb_dim]
        x = x + self.pos_emb[:,:,:x.size(1)]
        x = self.transformer(x)                   # [B,256+H,emb_dim]
        return self.head(x)                       # [B,256+H,256]

# —————— 训练 LDM ——————
def train_ldm(ldm, loader, device, epochs=10, lr=2e-4, ckpt_path=None):
    ldm.to(device).train()
    opt = optim.Adam(ldm.parameters(), lr=lr)
    try:
        for ep in range(1, epochs+1):
            total = 0.0
            pbar = tqdm(loader, desc=f"LDM Epoch {ep}/{epochs}", ncols=100)
            for x_seq in pbar:
                x_seq = x_seq.to(device)
                opt.zero_grad()
                _, vq_loss = ldm(x_seq)
                vq_loss.backward()
                opt.step()
                total += vq_loss.item()
                pbar.set_postfix(VQ=f"{total/len(loader):.4f}")

            # 每个 epoch 结束后都保存一次
            if ckpt_path:
                torch.save(ldm.state_dict(), ckpt_path)
                print(f"💾 已保存 LDM checkpoint：{ckpt_path}")

        print("✅ LDM training completed!")
    except Exception as e:
        # 发生任何错误也先保存当前状态，方便下次续训
        if ckpt_path:
            torch.save(ldm.state_dict(), ckpt_path)
            print(f"⚠️ 训练中断，已保存断点：{ckpt_path}")
        raise  # 再抛出异常，让你看到具体报错

# —————— 训练预测器 ——————
def train_predictor(predictor, loader, device, epochs=10, lr=1e-4):
    predictor.to(device).train()
    opt = optim.Adam(predictor.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    for ep in range(1, epochs+1):
        total = 0.0
        pbar = tqdm(loader, desc=f"Pred Epoch {ep}/{epochs}", ncols=100)
        for idx0, idx1, z_flat in pbar:
            idx0, idx1, z_flat = idx0.to(device), idx1.to(device), z_flat.to(device)
            opt.zero_grad()
            logits = predictor(idx0, z_flat)      # [B,256+H,256]
            # 取前256位置预测下一帧
            log_t = logits[:,:256,:].reshape(-1,256)
            tgt = idx1.view(-1)
            loss = crit(log_t, tgt)
            loss.backward()
            opt.step()
            total += loss.item()
            pbar.set_postfix(Loss=f"{total/len(loader):.4f}")
        print(f"✓ Epoch {ep} avg_loss={total/len(loader):.4f}")
    return predictor

# —————— 可视化 下一帧预测 ——————
def show_prediction(root_dir, vqvae, ldm, predictor, device, timestamp,
                    save_dir='fig_ldm', max_n=4, H=4):
    os.makedirs(save_dir, exist_ok=True)
    vqvae.eval(); ldm.eval(); predictor.eval()
    run_dir = sorted(Path(root_dir).glob('run_*'))[0]
    frames = sorted(run_dir.glob('*.png'))
    selected = frames[:H+1]
    imgs_seq = [T.Resize((64,64))(Image.open(p).convert('RGB')) for p in selected]
    x_seq = torch.stack([T.ToTensor()(img) for img in imgs_seq], dim=1).unsqueeze(0).to(device)
    with torch.no_grad():
        idx0 = vqvae.encode(x_seq[:,:,0]).view(1,-1)
        z_q,_ = ldm(x_seq)
        z_flat = z_q.view(1,H,-1)
        logits = predictor(idx0, z_flat)
        log_t = logits[:,:256,:]
        idx_next = log_t.argmax(-1).view(1,16,16)
        x_pred = vqvae.decode(idx_next).cpu().clamp(0,1).squeeze(0)
        x_true= T.ToTensor()(imgs_seq[1]).clamp(0,1)

    fig, axes = plt.subplots(1,3,figsize=(9,3))
    axes[0].imshow(x_seq[0,:,0].permute(1,2,0)); axes[0].set_title("Input t"); axes[0].axis('off')
    axes[1].imshow(x_pred.permute(1,2,0)); axes[1].set_title("Predicted t+1"); axes[1].axis('off')
    axes[2].imshow(x_true.permute(1,2,0)); axes[2].set_title("True t+1"); axes[2].axis('off')
    plt.tight_layout()
    path = os.path.join(save_dir, f"pred_ldm_{timestamp}.png")
    plt.savefig(path, dpi=150); plt.close(fig)
    print(f"✅ LDM-enhanced prediction saved: {path}")

# —————— 主流程 ——————
if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    timestamp = time.strftime('%Y%m%d_%H%M%S')

    root_dir = '../../../mnt/sdc/zhongqirui/gameruns'
    H = 4

    # 1) 加载 & 鉴定 VQ-VAE
    vqvae = VQVAE().to(device)
    vqvae.load_state_dict(torch.load('weights/vqvae_20250607_155540.pth', map_location=device))

    # 2) 构建并训练 LDM
    raw_ds = SequenceRawDataset(root_dir, seq_len=H+1)
    raw_loader = DataLoader(raw_ds, batch_size=1, shuffle=True, num_workers=2)
    ldm = LatentDynamicsModel(H=H)
    checkpoint_path = 'weights/ldm_checkpoint.pth'
    os.makedirs('weights', exist_ok=True)
    if os.path.exists(checkpoint_path):
        print(f"🔄 检测到已有 LDM checkpoint，加载：{checkpoint_path}")
        ldm.load_state_dict(torch.load(checkpoint_path, map_location=device))
    else:
        print("🚀 从头开始训练 LDM")
        train_ldm(ldm, raw_loader, device, epochs=5, lr=2e-4, ckpt_path=checkpoint_path)

    # 3) 构建 Predictor 数据集 & 训练
    pred_ds = SequenceSeqDataset(root_dir, vqvae, ldm, device, seq_len=H+1)
    pred_loader = DataLoader(pred_ds, batch_size=64, shuffle=True, num_workers=2)
    predictor = TokenPredictorWithLDM(H=H, feat_dim=64)
    predictor = train_predictor(predictor, pred_loader, device, epochs=20, lr=1e-4)

    # 4) 保存 Predictor 权重
    os.makedirs('weights', exist_ok=True)
    torch.save(predictor.state_dict(), f'weights/predictor_ldm_{timestamp}.pth')
    print(f"💾 Predictor+LDM 权重已保存: weights/predictor_ldm_{timestamp}.pth")

    # 5) 可视化 LDM 增强的下一帧预测
    show_prediction(root_dir, vqvae, ldm, predictor, device, timestamp, save_dir='fig_ldm', max_n=1, H=H)
