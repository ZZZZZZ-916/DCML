import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, ConcatDataset
import os
from sklearn.metrics import accuracy_score
from model import NanoObserver2D # 引用 model.py

CONFIG = {
    'dataset_root': './MU_SSiD_Dataset',
    'resolutions': ['224x224', '227x227', '256x256', '299x299', '331x331'],
    'model_path': './QAC_MultiRes_Artifacts/multires_10subset_best.pth',
    'batch_size': 1024,
    'device': torch.device("cuda" if torch.cuda.is_available() else "cpu")
}

def get_eval_loaders():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    loaders = {}
    for split, folder in [('Train', '1. Training'), ('Test', '3. Testing')]:
        ds_list = [datasets.ImageFolder(os.path.join(CONFIG['dataset_root'], res, res, folder), transform) 
                   for res in CONFIG['resolutions'] if os.path.exists(os.path.join(CONFIG['dataset_root'], res, res, folder))]
        loaders[split] = DataLoader(ConcatDataset(ds_list), batch_size=CONFIG['batch_size'])
    return loaders

def main():
    loaders = get_eval_loaders()
    model = NanoObserver2D(latent_dim=16).to(CONFIG['device'])
    model.load_state_dict(torch.load(CONFIG['model_path']))
    model.eval()

    results = {}
    for name, loader in loaders.items():
        all_preds, all_labels = [], []
        with torch.no_grad():
            for imgs, labels in loader:
                z = model(imgs.to(CONFIG['device']))
                preds = torch.argmax(-torch.cdist(z, model.qac.attractors), dim=1)
                all_preds.extend(preds.cpu().numpy()); all_labels.extend(labels.numpy())
        results[name] = accuracy_score(all_labels, all_preds)
        print(f"📊 {name} Accuracy: {results[name]:.2%}")

    gap = results['Train'] - results['Test']
    print(f"\n📉 Generalization Gap: {gap:.2%}")
    if gap < 0.10: print("✅ STATUS: HEALTHY (Good Generalization)") # 理想目標 < 9.5% [4, 5]
    else: print("❌ STATUS: OVERFITTING")

if __name__ == '__main__':
    main()
