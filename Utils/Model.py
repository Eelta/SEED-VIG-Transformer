from torch import nn
import torch
from .Attention_RoPE import Attention
from .LDS_FDS import FDS


# 卷积块EEGNet：深度可分离时间-空间解耦卷积
class ConvBlock(nn.Module):
    def __init__(self, in_ch, length, seq_len, dim, drop_rate):  # (B, in_ch, length)
        super().__init__()
        stride = length // seq_len
        kernel = (stride * 2) | 1  # 时间卷积核

        # Depthwise 逐通道时间卷积（groups=in_ch，每个通道独立提取时序特征）
        self.time_conv = nn.Conv1d(in_channels=in_ch, out_channels=in_ch * 2, kernel_size=kernel, stride=stride,
                                   padding=kernel // 2, groups=in_ch, bias=False)  # (B, in_ch * 2, seq_len)
        self.bn_time = nn.BatchNorm1d(in_ch * 2)
        self.elu_time = nn.ELU()  # (B, in_ch * 2, seq_len)

        # Pointwise 跨通道混合（1×1 卷积，融合所有通道信息）
        self.channel_conv = nn.Conv1d(in_channels=in_ch * 2, out_channels=dim, kernel_size=1,
                                      bias=False)  # (B, dim, seq_len)
        self.bn_channel = nn.BatchNorm1d(dim)
        self.elu_channel = nn.ELU()

        self.dropout = nn.Dropout(drop_rate)  # (B, dim, seq_len)

    def forward(self, x):
        x = self.elu_time(self.bn_time(self.time_conv(x)))
        x = self.elu_channel(self.bn_channel(self.channel_conv(x)))
        x = self.dropout(x)
        return x


# Backbone模块
class Backbone(nn.Module):
    def __init__(self, eeg_ch, eog_ch, eeg_len, eog_len, seq_len, dim, drop_rate, p_mask, num_heads, max_seq_len):
        super().__init__()
        # 卷积特征提取
        self.conv_eeg = ConvBlock(eeg_ch, eeg_len, seq_len, dim, drop_rate)  # (B, dim, seq_len)
        self.conv_eog = ConvBlock(eog_ch, eog_len, seq_len, dim, drop_rate)  # (B, dim, seq_len)

        # 动态门控：根据输入特征自适应决定 EEG-EOG 融合比例
        self.gate_mlp = nn.Sequential(
            nn.Linear(dim * 2, dim // 4),
            nn.ELU(),
            nn.Linear(dim // 4, 1),
            nn.Sigmoid()
        )
        self.p_mask = p_mask

        # 归一化与注意力机制
        self.layer_norm = nn.LayerNorm(dim)
        self.mha = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.attention = Attention(dim=dim, n_heads=num_heads, max_seq_len=max_seq_len)

    def forward(self, eeg, eog):
        # 1. 维度变换与卷积，[B, seq_len, dim]
        eeg = self.conv_eeg(eeg.permute(0, 2, 1)).permute(0, 2, 1)
        eog = self.conv_eog(eog.permute(0, 2, 1)).permute(0, 2, 1)

        # 2. 动态门控：池化 → 拼接 → MLP → 逐样本融合权重
        eeg_pool = eeg.mean(dim=1)  # [B, dim]
        eog_pool = eog.mean(dim=1)  # [B, dim]
        alpha = self.gate_mlp(torch.cat([eeg_pool, eog_pool], dim=-1))  # [B, 1]

        # 3. 跨模态注意力（EEG 查询 EOG）+ 模态随机丢弃
        B, seq_len, dim = eeg.shape
        cross_out = torch.zeros_like(eeg)

        if self.training and self.p_mask > 0.0:
            keep_mask = torch.rand(B, device=eeg.device) > self.p_mask
            if keep_mask.any():
                eeg_sub = eeg[keep_mask]
                eog_sub = eog[keep_mask]
                sub_cross_out, _ = self.mha(query=eeg_sub, key=eog_sub, value=eog_sub)  # [B, seq_len, dim]
                # 使部分eog融合eeg，其余直接舍弃
                cross_out[keep_mask] = sub_cross_out
        else:
            cross_out, _ = self.mha(query=eeg, key=eog, value=eog)

        # 4. 逐样本自适应融合
        eeg_fused = self.layer_norm(eeg + alpha.unsqueeze(-1) * cross_out)  # [B, seq_len, dim]

        # 5. 自注意力
        eeg_out = self.attention(eeg_fused)  # [B, seq_len, dim]

        # 6. 全局平均池化，输出最终特征
        pooled_out = eeg_out.mean(dim=1)  # [B, dim]
        return pooled_out


# 主网络
class Net(nn.Module):
    def __init__(self,
                 eeg_ch,
                 eog_ch,
                 eeg_len,
                 eog_len,
                 seq_len,
                 dim,
                 drop_rate,
                 p_mask,
                 num_heads,
                 max_seq_len,
                 num_bins,
                 sigma):
        super().__init__()

        # 冻结/解冻的最小单元
        self.backbone = Backbone(
            eeg_ch, eog_ch, eeg_len, eog_len, seq_len, dim,
            drop_rate, p_mask, num_heads, max_seq_len
        )

        self.fds = FDS(feature_dim=dim, num_bins=num_bins, sigma=sigma)

        # 回归头
        self.reg_head = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.BatchNorm1d(dim // 2),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(dim // 2, dim // 4),
            nn.BatchNorm1d(dim // 4),
            nn.ReLU(),
            nn.Linear(dim // 4, 1),
        )

    def forward(self, eeg, eog, targets=None, bin_edges=None):
        # 提取高维特征
        features = self.backbone(eeg, eog)  # [B, dim]
        # 经 FDS 模块（内部根据需要决定是否校准）
        calibrated_features = self.fds(features, targets, bin_edges)
        # 回归预测
        result = self.reg_head(calibrated_features)  # [B, 1]
        return result
