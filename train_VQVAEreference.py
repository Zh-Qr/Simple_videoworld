import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from PIL import Image
import numpy as np
import os
import torch.nn.functional as F
from typing import Tuple
import time
from tqdm import tqdm

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
    
class TokenTransformer(nn.Module):
    """
    接收 [B,D,7,14]，展平成 98 个 token 序列，
    经过 L 层 TransformerEncoder 后再 reshape 回 [B,D,7,14]。
    """
    def __init__(self, emb_dim=64, num_layers=4, n_head=8):
        super().__init__()
        self.h, self.w = 7, 14                        # 固定网格
        self.seq_len   = self.h * self.w              # 98
        self.pos = nn.Parameter(torch.randn(1, self.seq_len, emb_dim))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim, nhead=n_head,
            batch_first=True, dim_feedforward=emb_dim*4)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers)

    def forward(self, z_q):                           # z_q: [B,D,7,14]
        B, D, H, W = z_q.shape                       # H=7, W=14
        tokens = z_q.flatten(2)                      # [B,D,98]
        tokens = tokens.permute(0,2,1)               # [B,98,D]
        x = tokens + self.pos                        # 加位置编码
        x = self.encoder(x)                          # Transformer
        x = x.permute(0,2,1).view(B, D, H, W)        # 还原 [B,D,7,14]
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
        self.tf = TokenTransformer(embedding_dim, num_layers=tf_layers)
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
    def __init__(self, img_npy, lab_npy, transform=None):
        self.images  = np.load(img_npy).astype(np.uint8)   # ensure uint8
        self.labels  = np.load(lab_npy).astype(np.uint8)
        self.t = transform

    def __len__(self): return len(self.images)

    def __getitem__(self, idx):
        img  = Image.fromarray(self.images[idx],  mode='L')
        lab  = Image.fromarray(self.labels[idx], mode='L')
        if self.t:
            img = self.t(img)
            lab = self.t(lab)
        return img, lab
    
def train_vqvae(model: VQVAE,
                dataloader: DataLoader,
                epochs: int = 10,
                lr: float   = 2e-4,
                device=None):

    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.99))

    for ep in range(1, epochs+1):
        tot = rec_sum = vq_sum = 0.0
        pbar = tqdm(dataloader, desc=f'Epoch {ep}/{epochs}', ncols=100)

        for i, (imgs, labs) in enumerate(pbar, 1):     # 从 1 开始计数
            imgs, labs = imgs.to(device), labs.to(device)

            opt.zero_grad()
            recon, vq_loss = model(imgs)
            rec_loss = F.mse_loss(recon, labs)
            loss = rec_loss + vq_loss
            loss.backward()
            opt.step()

            tot     += loss.item()
            rec_sum += rec_loss.item()
            vq_sum  += vq_loss.item()

            pbar.set_postfix(
                Loss = f'{tot/i:.4f}',
                Recon= f'{rec_sum/i:.4f}',
                VQ   = f'{vq_sum/i:.4f}'
            )

        print(f'✓ Epoch {ep}  AvgLoss={tot/len(dataloader):.4f} '
              f'Recon={rec_sum/len(dataloader):.4f}  VQ={vq_sum/len(dataloader):.4f}')

    print('✅ Training completed!')

def show_batch(loader, model, device, timestamp,
               save_dir='fig', max_n = 1):
    """
    将一个 batch 的 (输入, 重建, 标签) 全部画在同一张图并保存。
    ----------
    max_n : 只画前 max_n 张；None 则画整个 batch。
    """
    model.eval()

    with torch.no_grad():
        imgs, labs = next(iter(loader))                 # 一个 batch
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

    save_path = os.path.join(save_dir, f'result_ref_{timestamp}.png')
    plt.savefig(save_path, dpi=150)
    print(f'✅ batch 可视化已保存: {save_path}')
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Initialize the model, loss function and optimizer
model = VQVAE(num_embeddings=64, embedding_dim=64, hid=128).to(device)
criterion = nn.MSELoss()  # Mean Squared Error loss for image reconstruction
optimizer = optim.Adam(model.parameters(), lr=0.001)
beta = 1.0

if __name__ == '__main__':
    timestamp = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
    # Create dataset and dataloaders
    train_dataset = AdditionDataset(img_npy='../../../../mnt/sdc/zhongqirui/static_MNIST/images.npy', 
                                    lab_npy='../../../../mnt/sdc/zhongqirui/static_MNIST/labels.npy', 
                                    transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    print("data is set")

    train_vqvae(model=model, dataloader=train_loader, epochs=100)

    weight_path = f'weight/vqvae_{timestamp}.pth'
    torch.save(model.state_dict(), weight_path)
    show_batch(train_loader, model, device, timestamp,
                save_dir='fig', max_n=16)
