利用 PCA 将演化轨迹投影至 3D 空间。
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from model import NanoObserver2D

def plot_manifold(model, samples):
    """动态展示特征点汇聚至吸引子的过程 [35, 36]。"""
    model.eval()
    # 跟踪 50 步演化轨迹 [34, 37]
    trajectories = model.forward_trace(samples) 
    pca = PCA(n_components=3)
    # 将 16 维潜空间投射到 3 维 [34, 38]
    traj_3d = pca.fit_transform(trajectories.reshape(-1, 16))
    # 绘图逻辑：Attractors 标记为恒星，轨迹标记为流星 [35, 36]
