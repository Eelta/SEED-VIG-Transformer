from torch import nn
import torch
from .Attention_RoPE import Attention


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
    def __init__(self, eeg_ch, eog_ch, eeg_len, eog_len, seq_len, dim, drop_rate, num_heads, max_seq_len):
        super().__init__()
        # 卷积特征提取
        self.conv_eeg = ConvBlock(eeg_ch, eeg_len, seq_len, dim, drop_rate)  # (B, dim, seq_len)
        self.conv_eog = ConvBlock(eog_ch, eog_len, seq_len, dim, drop_rate)  # (B, dim, seq_len)

        # 动态门控
        self.gate_mlp = nn.Sequential(
            nn.Linear(dim * 2, dim // 4),
            nn.ELU(),
            nn.Linear(dim // 4, 1),
            nn.Sigmoid()
        )

        # Pre-LN：每个注意力子层前独立归一化
        self.ln_eeg = nn.LayerNorm(dim)
        self.ln_eog = nn.LayerNorm(dim)
        self.ln_attn = nn.LayerNorm(dim)
        # 最终归一化
        self.ln_final = nn.LayerNorm(dim)

        # 交叉注意力、自注意力机制
        self.cross_attn = Attention(dim=dim, n_heads=num_heads, max_seq_len=max_seq_len)
        self.self_attn = Attention(dim=dim, n_heads=num_heads, max_seq_len=max_seq_len)

    def forward(self, eeg, eog):
        # 1. 维度变换与卷积，[B, seq_len, dim]
        eeg = self.conv_eeg(eeg.permute(0, 2, 1)).permute(0, 2, 1)
        eog = self.conv_eog(eog.permute(0, 2, 1)).permute(0, 2, 1)

        # 2. 动态门控：时间步级融合权重，[B, seq_len, 1]
        alpha = self.gate_mlp(torch.cat([eeg, eog], dim=-1))

        # 3. 跨模态注意力 (Pre-LN + 残差)
        eeg_norm = self.ln_eeg(eeg)
        eog_norm = self.ln_eog(eog)
        cross_out = self.cross_attn(query=eeg_norm, key=eog_norm, value=eog_norm)
        eeg = eeg + alpha * cross_out  # [B, seq_len, dim]

        # 4. 自注意力 (Pre-LN + 残差)
        attn_in = self.ln_attn(eeg)
        eeg = eeg + self.self_attn(attn_in, attn_in, attn_in)  # [B, seq_len, dim]

        # 5. 全局平均池化，输出最终特征
        eeg_out = self.ln_final(eeg)  # 最终归一化
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
                 num_heads,
                 max_seq_len):
        super().__init__()

        self.backbone = Backbone(
            eeg_ch, eog_ch, eeg_len, eog_len, seq_len, dim,
            drop_rate, num_heads, max_seq_len
        )

        # 回归头
        self.reg_head = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(dim // 2, dim // 4),
            nn.ReLU(),
            nn.Linear(dim // 4, 1),
        )

    def forward(self, eeg, eog):
        features = self.backbone(eeg, eog)  # [B, dim]
        return self.reg_head(features)       # [B, 1]
