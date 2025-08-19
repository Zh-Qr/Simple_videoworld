import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import os
import random

class AdditionDataset(Dataset):
    def __init__(self, root, transform=None, num_samples=10000):
        self.mnist = datasets.MNIST(root, train=True, download=True, transform=None)
        self.transform = transform
        self.digits = list(range(5))
        self.num_samples = num_samples
        
        self.digit_data = {i: [] for i in range(10)}
        for img, label in self.mnist:
            self.digit_data[label].append(img)
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        d1, d2 = random.sample(self.digits, 2)
        img1 = random.choice(self.digit_data[d1])
        img2 = random.choice(self.digit_data[d2])
        
        # 拼接
        w, h = img1.width + img2.width, img1.height
        combined = Image.new('L', (w, h))
        combined.paste(img1, (0,0))
        combined.paste(img2, (img1.width,0))
        
        # 标签图像和数值
        sum_val = d1 + d2
        sum_img = random.choice(self.digit_data[sum_val])
        
        if self.transform:
            combined = self.transform(combined)
            sum_img  = self.transform(sum_img)
        
        return combined, sum_img, sum_val


def save_dataset(loader, num_samples=10000, output_dir='./generated_data'):
    os.makedirs(output_dir, exist_ok=True)
    imgs, labs_img, labs_val = [], [], []
    
    for i, (img, lab_img, lab_val) in enumerate(loader):
        imgs.append    ( img .squeeze().numpy() )
        labs_img.append( lab_img.squeeze().numpy() )
        labs_val.append( int(lab_val) )
        
        if i >= num_samples-1:
            break
    
    imgs     = np.stack(imgs)       # shape (N,1,H,W) → (N,H,W)
    labs_img = np.stack(labs_img)
    labs_val = np.array(labs_val, dtype=np.int64)
    
    np.save(os.path.join(output_dir, 'images.npy'), imgs)
    np.save(os.path.join(output_dir, 'labels.npy'), labs_img)
    np.save(os.path.join(output_dir, 'values.npy'), labs_val)
    print(f"Saved {len(imgs)} samples to {output_dir}")


if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    ds = AdditionDataset(root='../../../../mnt/sdc/zhongqirui/MNIST', transform=transform, num_samples=50000)
    loader = DataLoader(ds, batch_size=1, shuffle=True)
    save_dataset(loader, num_samples=50000, output_dir='../../../../mnt/sdc/zhongqirui/static_MNIST')
