# DCML

## 🚀 项目简介
DCML 是一个将深度学习与非线性物理方程（Allen-Cahn 方程）深度融合的图像分类框架。该模型基于流形假设 (Manifold Hypothesis)，通过物理演化机制将高维视觉特征引导至低维物理流形上的吸引子盆地，旨在解决科学观测数据中的严重过拟合问题。

---

## 🧠 核心数学原理

1. **流形映射**: 使用 NanoObserver2D (轻量化 CNN) 将输入图像映射到 16 维潜空间 (Latent Space)。
2. **Allen-Cahn 动力学**: 特征在物理层 SpatialQAC 中经历 50 次迭代演化。

$$
\frac{\partial z}{\partial t} = D \nabla^2 z + f(z)
$$

* **扩散 (Diffusion)**: 利用拉普拉斯卷积核捕获空间相干性。
* **反应 (Reaction)**: 公式为 $diff \times (1.0 - diff^2)$，将特征推向对应的物理吸引子。

3. **量子噪声正则化**: 训练期间根据位能注入动态噪声，模拟量子涨落以增强泛化力。

---

## 📊 性能评估
实验表明，标准 CNN 在科学数据集上存在严重过拟合（泛化差距 >**20%**），而引入物理机制的 DCML 能将差距缩小至 **6.5% - 9.9%**。

---

## 🛠️ 快速開始 (Quick Start)

### 1. 環境配置 (Environment Setup)
本項目基於 PyTorch 框架開發，建議使用虛擬環境以確保依賴項隔離。

```bash
# 創建並啟動虛擬環境
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# 安裝核心依賴項
pip install -r requirements.txt

### 2. 數據準備 (Data Preparation)
由於 MU_SSiD 數據集體積較大，本倉庫不包含原始影像數據。請依照以下步驟準備：

1. **下載數據**: 請前往 [點此插入你的 Google Drive/Kaggle 鏈接] 下載完整數據集。
2. **解壓與排列**: 將下載的數據解壓至項目根目錄，並確保文件夾層級嚴格符合以下結構：

```text
./MU_SSiD_Dataset/
├── 224x224/
│   ├── 1. Training/
│   ├── 2. Validation/
│   └── 3. Testing/
├── 227x227/ ... 
└── 331x331/
