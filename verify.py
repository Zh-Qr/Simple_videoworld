import os
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import PILToTensor

# ---------------- 1) 设备 ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ---------------- 2) 自定义 MNISTUint8 ----------------
class MNISTUint8(datasets.MNIST):
    """
    直接使用 .data/.targets（uint8）构造 PIL，再做 transform。
    """
    def __init__(self, root, train=True, download=False, transform=None):
        # 不让父类做 transform
        super().__init__(root, train=train, download=download, transform=None)
        self.my_transform = transform

    def __getitem__(self, idx):
        # 1) 从 self.data/self.targets 取 uint8 图、label
        arr = self.data[idx].numpy().astype(np.uint8)   # (28,28), uint8
        label = int(self.targets[idx].item())
        # 2) 转回 PIL
        pil = Image.fromarray(arr, mode='L')
        # 3) 由你的 transform（PILToTensor→Normalize）生成 final tensor
        if self.my_transform:
            img = self.my_transform(pil)
        else:
            img = pil
        return img, label

# ---------------- 3) Transform：PIL→uint8 Tensor→[-1,1] Float ----------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# ---------------- 4) 划分数据 ----------------
root = '../../../../mnt/sdc/zhongqirui/MNIST'
full_ds = MNISTUint8(root, train=True, download=True, transform=transform)

torch.manual_seed(42)
n_train = int(len(full_ds) * 0.8)
n_val   = len(full_ds) - n_train
train_ds, val_ds = random_split(full_ds, [n_train, n_val])

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True,  drop_last=True)
val_loader   = DataLoader(val_ds,   batch_size=64, shuffle=False)

print(f"Dataset split → Train {n_train}  |  Val {n_val}")

# ---------------- 5) 定义网络 ----------------
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.pool  = nn.MaxPool2d(2, 2)
        self.fc1   = nn.Linear(64 * 7 * 7, 128)
        self.fc2   = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

model     = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# ---------------- 6) 训练 + 验证 ----------------
def fit(model, train_ld, val_ld, epochs=20):
    best_acc = 0.0
    for ep in range(1, epochs+1):
        # ———— 训练 ————
        model.train()
        running = 0.0
        pbar = tqdm(train_ld, desc=f'Epoch {ep}/{epochs}', ncols=100)
        for i, (imgs, lbls) in enumerate(pbar, 1):
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), lbls)
            loss.backward()
            optimizer.step()
            running += loss.item()
            pbar.set_postfix(train_loss=f'{running/i:.4f}')

        # ———— 验证 ————
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for imgs, lbls in val_ld:
                imgs, lbls = imgs.to(device), lbls.to(device)
                preds = model(imgs).argmax(1)
                total   += lbls.size(0)
                correct += (preds == lbls).sum().item()
        acc = 100 * correct / total
        print(f'  ↳ val_acc = {acc:.2f}%')

        if acc > best_acc:
            best_acc = acc
            os.makedirs('weight', exist_ok=True)
            torch.save(model.state_dict(), 'weight/MNIST_best.pth')
            print('  ✅ New best model saved')

    print(f'Finished. Best Val Acc = {best_acc:.2f}%')

if __name__ == '__main__':
    fit(model, train_loader, val_loader, epochs=5)