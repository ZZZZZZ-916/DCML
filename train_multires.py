import torch
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import datasets, transforms
import os
from model import NanoObserver2D
from tqdm import tqdm

CONFIG = {
    'dataset_root': './MU_SSiD_Dataset',
    'resolutions': ['224x224', '227x227', '256x256', '299x299', '331x331'],
    'save_dir': './QAC_MultiRes_Artifacts',
    'batch_size': 1024,
    'epochs': 30,
    'lr': 1e-3,
    'device': torch.device("cuda" if torch.cuda.is_available() else "cpu")
}

def get_multires_loaders():
    """扫描并合并多分辨率数据集 [17, 18]。"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    train_ds, val_ds = [], []
    for res in CONFIG['resolutions']:
        base = os.path.join(CONFIG['dataset_root'], res, res) if os.path.exists(os.path.join(CONFIG['dataset_root'], res, res)) else os.path.join(CONFIG['dataset_root'], res)
        train_ds.append(datasets.ImageFolder(os.path.join(base, '1. Training'), transform=transform))
        val_ds.append(datasets.ImageFolder(os.path.join(base, '2. Validation'), transform=transform))
    return (DataLoader(ConcatDataset(train_ds), batch_size=CONFIG['batch_size'], shuffle=True),
            DataLoader(ConcatDataset(val_ds), batch_size=CONFIG['batch_size']))

def train():
    os.makedirs(CONFIG['save_dir'], exist_ok=True)
    train_loader, val_loader = get_multires_loaders()
    model = NanoObserver2D().to(CONFIG['device'])
    # 使用 AdamW 优化器 [19, 20]
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=1e-2)
    best_acc = 0.0
    for epoch in range(CONFIG['epochs']):
        model.train()
        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            imgs, labels = imgs.to(CONFIG['device']), labels.to(CONFIG['device'])
            optimizer.zero_grad()
            # 基于距离的度量学习损失 [21, 22]
            z_vec = model(imgs)
            loss = F.cross_entropy(-torch.cdist(z_vec, model.qac.attractors), labels)
            loss.backward()
            optimizer.step()
        # 验证并保存最佳模型 [23, 24]
