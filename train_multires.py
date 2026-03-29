import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, ConcatDataset, Subset
import os
from tqdm import tqdm
from model import NanoObserver2D # 確保 model.py 在同一目錄下

# 1. 配置信息 [2, 3]
CONFIG = {
    'dataset_root': './MU_SSiD_Dataset',
    'resolutions': ['224x224', '227x227', '256x256', '299x299', '331x331'],
    'save_dir': './QAC_MultiRes_Artifacts',
    'batch_size': 1024,
    'epochs': 30,
    'lr': 1e-3,
    'latent_dim': 16,
    'device': torch.device("cuda" if torch.cuda.is_available() else "cpu")
}

def get_multires_loaders():
    """掃描並實現 10% 子集截取 [1, 4]。"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    train_ds_list, val_ds_list = [], []
    for res in CONFIG['resolutions']:
        base = os.path.join(CONFIG['dataset_root'], res, res) if os.path.exists(os.path.join(CONFIG['dataset_root'], res, res)) else os.path.join(CONFIG['dataset_root'], res)
        train_path = os.path.join(base, '1. Training')
        val_path = os.path.join(base, '2. Validation')
        if os.path.exists(train_path):
            train_ds_list.append(datasets.ImageFolder(train_path, transform=transform))
            val_ds_list.append(datasets.ImageFolder(val_path, transform=transform))

    # 合併全量數據
    full_train_ds = ConcatDataset(train_ds_list)
    
    # === 🌟 任務一核心修改：隨機抽取 10% 數據 (約 36,000 張) ===
    subset_size = int(0.1 * len(full_train_ds)) 
    indices = torch.randperm(len(full_train_ds))[:subset_size]
    train_subset = Subset(full_train_ds, indices)
    # ========================================================

    return (DataLoader(train_subset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=8, pin_memory=True),
            DataLoader(ConcatDataset(val_ds_list), batch_size=CONFIG['batch_size'], shuffle=False, num_workers=8, pin_memory=True))

def train():
    os.makedirs(CONFIG['save_dir'], exist_ok=True)
    train_loader, val_loader = get_multires_loaders()
    
    model = NanoObserver2D(latent_dim=CONFIG['latent_dim']).to(CONFIG['device'])
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=1e-2)
    scaler = torch.amp.GradScaler('cuda') # 提升訓練速度 [5]

    for epoch in range(CONFIG['epochs']):
        model.train()
        pbar = tqdm(train_loader, desc=f"Ep {epoch+1}")
        for imgs, labels in pbar:
            imgs, labels = imgs.to(CONFIG['device']), labels.to(CONFIG['device'])
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                z_vec = model(imgs)
                # 使用度量學習損失：負距離作為 logits [6, 7]
                logits = -torch.cdist(z_vec, model.qac.attractors)
                loss = F.cross_entropy(logits, labels)
            
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # 梯度裁剪保證物理層穩定 [6]
            scaler.step(optimizer)
            scaler.update()

        print(f"Epoch {epoch+1} 訓練完成，權重已保存至 {CONFIG['save_dir']}")
        torch.save(model.state_dict(), os.path.join(CONFIG['save_dir'], 'multires_10subset_best.pth'))

if __name__ == '__main__':
    train()
