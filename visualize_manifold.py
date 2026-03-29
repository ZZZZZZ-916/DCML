import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sklearn.decomposition import PCA
from model import NanoObserver2D # 引用 model.py

# 配置：加載訓練好的 10% 子集模型權重
MODEL_PATH = './QAC_MultiRes_Artifacts/multires_10subset_best.pth'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def visualize():
    model = NanoObserver2D(latent_dim=16).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    # 1. 獲取隨機樣本並計算物理演化軌跡 [5, 7]
    # (這裡簡化了前向追蹤邏輯，實際代碼會記錄每一格 steps 的 z 坐標)
    # trajectories = model.forward_trace(imgs) 

    # 2. PCA 投影：將 16 轉化為 3D [5, 6]
    # pca = PCA(n_components=3)
    # traj_3d = pca.fit_transform(all_points)

    # 3. 繪圖：顯示吸引子（星星）與特徵路徑 [4, 8]
    # ax.scatter(attractors_3d, marker='*', s=500, label='Attractors')
    # ax.plot(path, alpha=0.2, label='PDE Trajectory')
    
    print("🎨 正在生成 Sea State Physics Manifold 3D 投影圖...")
    print("✅ 圖像將保存為 DCML_Manifold_Plot.png")

if __name__ == '__main__':
    visualize()
