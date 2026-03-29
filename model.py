import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialQAC(nn.Module):
    """实现 50 步 Allen-Cahn 动力学演化 [5, 6]。"""
    def __init__(self, latent_dim=16, num_classes=4, steps=50):
        super().__init__()
        self.steps = steps
        self.dt = 0.05
        self.diffusion_rate = 0.1
        # 注册 4 个正交吸引子 [3, 7]
        self.register_buffer('attractors', torch.eye(num_classes, latent_dim))
        # 注册拉普拉斯卷积核 [3, 8]
        laplacian = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32)
        self.register_buffer('laplacian_kernel', laplacian.view(1, 1, 3, 3).repeat(latent_dim, 1, 1, 1))

    def get_potential(self, z):
        """计算特征点的位能 V [7, 9]。"""
        b, c, h, w = z.shape
        z_perm = z.permute(0, 2, 3, 1).reshape(-1, c)
        dists = torch.cdist(z_perm, self.attractors)
        min_v, _ = torch.min(dists.pow(2), dim=1)
        return min_v.view(b, 1, h, w)

    def forward(self, z):
        z = torch.clamp(z, -2.0, 2.0)
        # 训练模式：注入量子噪声 [10, 11]
        if self.training:
            V = self.get_potential(z)
            current_beta = 10.0 / (1.0 + 10.0 * torch.clamp(V, max=10.0))
            sigma = torch.clamp(torch.rsqrt(2.0 * current_beta + 1e-6), max=0.5)
            z = z + torch.randn_like(z) * sigma
        # 物理演化循环 [8, 11]
        for _ in range(self.steps):
            laplacian = F.conv2d(z, self.laplacian_kernel, padding=1, groups=z.shape[1])
            b, c, h, w = z.shape
            z_flat = z.permute(0, 2, 3, 1).reshape(-1, c)
            weights = F.softmax(-torch.cdist(z_flat, self.attractors) * 10.0, dim=1)
            target = torch.matmul(weights, self.attractors).view(b, h, w, c).permute(0, 3, 1, 2)
            # Allen-Cahn 反应项与扩散项结合 [12, 13]
            reaction = (target - z) * (1.0 - (target - z).pow(2))
            z = torch.clamp(z + self.dt * (self.diffusion_rate * laplacian + reaction), -3.0, 3.0)
        return z

class NanoObserver2D(nn.Module):
    """20k 参数量的轻量化 CNN Backbone [4, 14, 15]。"""
    def __init__(self, latent_dim=16):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 24, 3, 2, 1), nn.BatchNorm2d(24), nn.ReLU(),
            nn.Conv2d(24, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, latent_dim, 3, 2, 1), nn.BatchNorm2d(latent_dim), nn.Tanh()
        )
        self.qac = SpatialQAC(latent_dim=latent_dim)

    def forward(self, x):
        z = self.qac(self.features(x))
        return F.adaptive_avg_pool2d(z, (1, 1)).view(x.size(0), -1)
