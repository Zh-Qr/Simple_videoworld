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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
    
# Data loading and preprocessing
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Convert to grayscale to match input channels
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1] range
])

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
    
# =========================================================
# --- 子函数 1：单 Epoch 训练 ------------------------------
# =========================================================
perc_net = lpips.LPIPS(net='alex').to(device)
def train_one_epoch(model,
                    loader,
                    optimizer,
                    device,
                    perc,
                    scaler=None,
                    epoch_id = 1,
                    total_epochs = 10):
    """
    训练一个 epoch 并实时显示：
    ┌ Epoch 3/30 ── 100%|████████| 500/500 [00:12<00:00, 40.08it/s] Loss=0.0193 Recon=0.0121 VQ=0.0072 ┐
    """
    model.train()
    tot = rec_sum = vq_sum = perc_sum = 0.0

    desc = (f'Epoch {epoch_id}/{total_epochs}' if epoch_id else 'Train')
    pbar = tqdm(loader, desc=desc, ncols=100)

    for i, (imgs, labs, _) in enumerate(pbar, 1):
        imgs, labs = imgs.to(device), labs.to(device)

        optimizer.zero_grad(set_to_none=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type='cuda', enabled=scaler is not None):
            recon, vq_loss = model(imgs)
            rec_loss  = F.mse_loss(recon, labs)

            # ----- LPIPS -----
            recon_3 = recon.repeat(1,3,1,1)
            labs_3  = labs .repeat(1,3,1,1)
            recon_lp = F.interpolate(recon_3, 64, mode='bilinear', align_corners=False)
            labs_lp  = F.interpolate(labs_3 , 64, mode='bilinear', align_corners=False)
            perc_loss = perc_net(recon_lp, labs_lp).mean()

            loss = (1-perc) * rec_loss + vq_loss + perc * perc_loss

        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        tot     += loss.item()
        rec_sum += rec_loss.item()
        vq_sum  += vq_loss.item()
        perc_sum+= perc_loss.item()

        # 实时显示当前 batch 的平均指标
        pbar.set_postfix(
            Loss = f'{tot/i:.4f}',
            Recon= f'{rec_sum/i:.4f}',
            Perc = f'{perc_sum/i:.4f}',
            VQ   = f'{vq_sum/i:.4f}'
        )

    n = len(loader)
    return tot/n, rec_sum/n, vq_sum/n


# =========================================================
# --- 子函数 2：测试 --------------------------------------
# =========================================================
@torch.no_grad()
def evaluate(model, verify_model, loader, device):
    """
    • recon_mse: VQ‐VAE 重建误差（每像素 MSE）
    • cls_acc:  CNN 对重建图的分类准确率
    """
    model.eval()
    verify_model.eval()

    mse = torch.nn.MSELoss(reduction='sum')
    se_sum = 0.0
    cls_correct = 0
    total = 0

    for imgs, labs, vals in loader:
        imgs, labs, vals  = imgs.to(device), labs.to(device), vals.to(device)

        # 1) VQ-VAE 重建
        recon, _ = model(imgs)                  # recon ∈ [-1,1]（Tanh）或 [0,1]（Sigmoid）

        # 2) 先反归一化到 [0,1]
        recons_denorm = (recon * 0.5 + 0.5).clamp(0, 1)
        labs_denorm   = (labs  * 0.5 + 0.5).clamp(0, 1)

        # 3) 再按 CNN 训练时的 (x-0.5)/0.5 映射到 [-1,1]
        recons_for_cnn = (recons_denorm - 0.5) / 0.5
        labs_for_cnn   = (labs_denorm   - 0.5) / 0.5

        # 4) CNN 分类
        logits_recon  = verify_model(-recons_for_cnn)
        logits_target = verify_model(-labs_for_cnn)
        preds_recon   = logits_recon .argmax(dim=1)
        preds_target  = logits_target.argmax(dim=1)

        cls_correct += (preds_recon == vals).sum().item()
        total       += preds_target.numel()

        # 5) 累加重建 MSE
        se_sum += mse(recon, labs).item()

    # 每像素 MSE
    pixel_count = total * labs.shape[2] * labs.shape[3] / preds_target.numel()
    recon_mse = se_sum / pixel_count
    cls_acc   = cls_correct / total

    return recon_mse, cls_acc

# =========================================================
# --- 主控函数：训练多个 Epoch 并在每 Epoch 后评测 ---------
# =========================================================
def start_train(timestamp,
                model,
                verify_model,
                train_loader,
                test_loader,
                perc,
                epochs          = 20,
                lr              = 2e-4,
                save_dir        = 'weight',
                use_fp16        = False,
                device          = None):

    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # ---------- 准备目录 ----------
    os.makedirs(save_dir, exist_ok=True)
    run_log_dir = 'log'
    os.makedirs(run_log_dir, exist_ok=True)
    csv_path = os.path.join(run_log_dir, f'train_{timestamp}.csv')

    # ---------- CSV 初始化 ----------
    csv_f  = open(csv_path, 'w', newline='')
    writer = csv.writer(csv_f)
    writer.writerow(['epoch', 'train_loss', 'train_recon',
                     'train_vq', 'test_mse', 'cls_acc'])
    csv_f.flush()
    print(f'📝 CSV log → {csv_path}')

    model       = model.to(device)
    verify_model= verify_model.to(device).eval()      # CNN 只推断
    optimizer   = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9,0.99))
    scaler      = torch.cuda.amp.GradScaler() if (use_fp16 and device.type=='cuda') else None

    best_acc = -1
    for ep in range(1, epochs+1):
        train_loss, train_rec, train_vq = train_one_epoch(
            model, train_loader, optimizer, device, perc, scaler, epoch_id=ep, total_epochs=epochs)

        test_mse, test_acc = evaluate(model, verify_model, test_loader, device)

        print(f'[Epoch {ep:03d}/{epochs}] '
              f'Loss={train_loss:.4f} | Recon={train_rec:.4f} | VQ={train_vq:.4f} || '
              f'TestMSE={test_mse:.5f} | ClsAcc={test_acc*100:.2f}%')
        
        writer.writerow([ep, train_loss, train_rec, train_vq, test_mse, test_acc])
        csv_f.flush()                      # 实时落盘

        # 按最优分类准确率保存
        if test_acc > best_acc:
            best_acc = test_acc
            best_path = os.path.join(save_dir, f'vqvae_best_{timestamp}.pth')
            torch.save(model.state_dict(), best_path)
            print(f'  ✅ Saved new best model to {best_path}')

        torch.cuda.empty_cache()  # 释放无用显存

    print(f'🎉 Training finished. Best ClsAcc = {best_acc*100:.2f}%')





def show_batch(loader, model, device, timestamp,
               save_dir='fig', max_n = 1, label = ''):
    """
    将一个 batch 的 (输入, 重建, 标签) 全部画在同一张图并保存。
    ----------
    max_n : 只画前 max_n 张；None 则画整个 batch。
    """
    model.eval()

    with torch.no_grad():
        imgs, labs, _ = next(iter(loader))                 # 一个 batch
        imgs, labs = imgs.to(device), labs.to(device)
        recons, _  = model(imgs)

    # ---- 反归一化并搬到 CPU ----
    imgs   = (imgs.cpu()  * 0.5 + 0.5).clamp(0,1)
    recons = (recons.cpu()*0.5 + 0.5).clamp(0,1)
    labs   = (labs.cpu()  *0.5 + 0.5).clamp(0,1)

    B = imgs.size(0) if max_n is None else min(max_n, imgs.size(0))
    ncols, nrows = 3, B                                  # 每行 Input / Recon / Label

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(3*ncols, 3*nrows),
                             squeeze=False)

    titles = ['Input', 'Recon', 'Label']
    for r in range(nrows):
        axes[r][0].imshow(imgs  [r,0], cmap='gray')
        axes[r][1].imshow(recons[r,0], cmap='gray')
        axes[r][2].imshow(labs  [r,0], cmap='gray')
        for c in range(ncols):
            axes[r][c].axis('off')
            if r == 0: axes[r][c].set_title(titles[c])

    save_path = os.path.join(save_dir, f'result_{label}_{timestamp}.png')
    plt.savefig(save_path, dpi=150)
    print(f'✅ batch 可视化已保存: {save_path}')

    

# Initialize the model, loss function and optimizer
model = VQVAE(num_embeddings=64, embedding_dim=64, hid=128).to(device)
criterion = nn.MSELoss()  # Mean Squared Error loss for image reconstruction
optimizer = optim.Adam(model.parameters(), lr=0.001)
beta = 1.0

# ------- 改进写法 -------
state_dict = torch.load('weight/MNIST.pth',
                        map_location='cpu',
                        weights_only=True)
verify_model = SimpleCNN()
verify_model.load_state_dict(state_dict )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--perc',   type=float, default=0.20,
                        help='λ_perc：LPIPS 感知损失权重')
    parser.add_argument('--epochs', type=int,   default=100)
    parser.add_argument('--batch',  type=int,   default=32)
    args = parser.parse_args()
    # reproducible split
    torch.manual_seed(42)

    timestamp = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
    print('Loading numpy files …')

    # -------- 1) 先整体加载完整数据集 --------
    full_ds = AdditionDataset(
        img_npy='../../../../mnt/sdc/zhongqirui/static_MNIST/images.npy',
        lab_npy='../../../../mnt/sdc/zhongqirui/static_MNIST/labels.npy',
        val_npy='../../../../mnt/sdc/zhongqirui/static_MNIST/values.npy',
        transform=transform)

    # -------- 2) 按 8:2 随机划分 (train / test) --------
    total_len   = len(full_ds)
    train_len   = int(0.8 * total_len)
    test_len    = total_len - train_len
    train_ds, test_ds = random_split(full_ds, [train_len, test_len])

    # -------- 3) 构建 DataLoader --------
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True,  drop_last=True)
    test_loader  = DataLoader(test_ds,  batch_size=64, shuffle=False, drop_last=False)

    print(f"Dataset loaded ✓  Train: {len(train_ds)}  Test: {len(test_ds)}")

    # -------- 4) 训练 VQ‑VAE 并评估 --------
    start_train(timestamp, model, verify_model,
                train_loader=train_loader,
                test_loader=test_loader,
                perc=args.perc,
                epochs=100,
                lr=2e-4,
                save_dir='weight',
                use_fp16=True)

    # -------- 5) 额外保存一次当前权重、可视化 --------
    os.makedirs('weight', exist_ok=True)
    torch.save(model.state_dict(), f'weight/vqvae_{timestamp}.pth')

    os.makedirs('fig', exist_ok=True)
    show_batch(test_loader, model, device, timestamp,
               save_dir='fig', max_n=16, label='ref')
    
    show_batch(train_loader, model, device, timestamp,
               save_dir='fig', max_n=16, label='train')