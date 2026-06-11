import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d


# ==========================================
# LDS (Label Distribution Smoothing) 工具
# ==========================================
def get_lds_weights(train_loader, num_bins, sigma):
    # 遍历原生 Dataset 获取标签
    y_data = np.array([label.cpu().numpy() for _, _, label in train_loader.dataset])
    """
    计算基于核平滑的连续标签权重
    """
    # 1. 划分连续空间并统计经验分布 (直方图)
    counts, bin_edges = np.histogram(y_data, bins=num_bins)  # 返回 频数数组，区间边界数组
    # 2. 使用高斯核进行分布平滑
    smoothed_counts = gaussian_filter1d(counts, sigma=sigma)
    # 3. 计算权重 (密度倒数)，并截断极小值防止爆炸
    smoothed_counts = np.clip(smoothed_counts, a_min=1e-5, a_max=None)
    weights = 1.0 / smoothed_counts
    # 4. 权重归一化
    weights = weights / np.sum(weights) * num_bins
    return torch.FloatTensor(weights), torch.FloatTensor(bin_edges)


def get_bin_idx(y, bin_edges):
    bin_edges = bin_edges.to(y.device)
    """辅助函数：将连续目标值映射到对应的 Bin 索引"""
    idx = torch.bucketize(y, bin_edges) - 1
    # 确保索引不越界
    idx = torch.clamp(idx, 0, len(bin_edges) - 2)
    return idx


# ==========================================
# FDS (Feature Distribution Smoothing) 模块
# ==========================================
class FDS(nn.Module):
    def __init__(self, feature_dim, num_bins, sigma):
        super(FDS, self).__init__()
        self.feature_dim = feature_dim
        self.num_bins = num_bins
        self.sigma = sigma
        # 注册不需要梯度更新的缓冲区，用于存储各个 Bin 的统计量
        self.register_buffer('running_mean', torch.zeros(num_bins, feature_dim))
        self.register_buffer('running_var', torch.ones(num_bins, feature_dim))
        self.register_buffer('smoothed_mean', torch.zeros(num_bins, feature_dim))
        self.register_buffer('smoothed_var', torch.ones(num_bins, feature_dim))
        # 校准融合系数：1.0 = 完全使用平滑后特征，0.0 = 原始特征
        # 由 train_s2 根据 batch 进度动态更新，实现平滑过渡
        self.alpha = 1.0

    def update_and_smooth_stats(self, features, targets, bin_edges):
        """在 Stage 1 结束时被调用：统计所有特征，并平滑"""
        device = features.device
        targets = targets.squeeze()
        bins = get_bin_idx(targets, bin_edges.to(device))
        # 统计每个 Bin 的均值和方差
        for b in range(self.num_bins):
            mask = (bins == b)
            count = mask.sum()
            if count > 1:
                bin_feats = features[mask]
                self.running_mean[b] = bin_feats.mean(dim=0)
                self.running_var[b] = bin_feats.var(dim=0, correction=0)
            else:
                # 样本不足（0 或 1）：标记为 NaN，后续插值填充
                self.running_mean[b] = float('nan')
                self.running_var[b] = float('nan')
        # 使用高斯核进行特征空间平滑
        mean_np = self.running_mean.cpu().numpy()
        var_np = self.running_var.cpu().numpy()
        # 对样本不足的 bin 进行线性插值填充，避免污染平滑结果
        nan_mask = np.isnan(mean_np)
        if nan_mask.any():
            mean_np = pd.DataFrame(mean_np).interpolate(limit_direction='both').to_numpy()
            var_np = pd.DataFrame(var_np).interpolate(limit_direction='both').to_numpy()
            # 修复 running_mean/running_var 中的 NaN，避免 Stage 2 forward 时报错
            self.running_mean.copy_(torch.from_numpy(mean_np).to(device))
            self.running_var.copy_(torch.from_numpy(var_np).to(device))
        smooth_mean = gaussian_filter1d(mean_np, sigma=self.sigma, axis=0)
        smooth_var = gaussian_filter1d(var_np, sigma=self.sigma, axis=0)
        self.smoothed_mean.copy_(torch.from_numpy(smooth_mean).to(device))
        self.smoothed_var.copy_(torch.from_numpy(smooth_var).to(device))

    def forward(self, x, targets=None, bin_edges=None):
        # 仅在提供 targets/bin_edges 时（Stage 2 训练）执行特征校准
        if targets is not None and bin_edges is not None:
            device = x.device
            targets = targets.squeeze()
            bins = get_bin_idx(targets, bin_edges.to(device))
            # 获取原始统计量和平滑后的统计量
            mu = self.running_mean[bins]
            var = self.running_var[bins]
            smooth_mu = self.smoothed_mean[bins]
            smooth_var = self.smoothed_var[bins]
            # 特征校准公式：消除原始方差，注入平滑后的方差与均值
            x_calibrated = (x - mu) / torch.sqrt(var + 1e-5)
            x_calibrated = x_calibrated * torch.sqrt(smooth_var + 1e-5) + smooth_mu
            # Alpha 融合：前 80% batch α=1.0，后 20% 线性衰减到 0.0
            # 让 reg_head 从平滑特征平稳过渡回原始特征分布
            alpha = getattr(self, 'alpha', 1.0)
            if alpha >= 1.0:
                return x_calibrated
            elif alpha <= 0.0:
                return x
            else:
                return alpha * x_calibrated + (1 - alpha) * x
        return x
