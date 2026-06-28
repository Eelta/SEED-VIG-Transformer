import torch
from torch import nn
from .Model import ConvBlock, Backbone
from .Attention_RoPE import Attention


class BackboneEEGOnly(nn.Module):
    """仅 EEG 通道：无 EOG 卷积、无跨模态注意力、无门控"""

    def __init__(self, eeg_ch, eeg_len, seq_len, dim, drop_rate, num_heads, max_seq_len):
        super().__init__()
        self.conv_eeg = ConvBlock(eeg_ch, eeg_len, seq_len, dim, drop_rate)
        self.ln_attn = nn.LayerNorm(dim)
        self.ln_final = nn.LayerNorm(dim)
        self.self_attn = Attention(dim=dim, n_heads=num_heads, max_seq_len=max_seq_len)

    def forward(self, eeg, eog):
        eeg = self.conv_eeg(eeg.permute(0, 2, 1)).permute(0, 2, 1)
        attn_in = self.ln_attn(eeg)
        eeg = eeg + self.self_attn(attn_in, attn_in, attn_in)
        eeg_out = self.ln_final(eeg)
        return eeg_out.mean(dim=1)


class BackboneEOGOnly(nn.Module):
    """仅 EOG 通道：无 EEG 卷积、无跨模态注意力、无门控"""

    def __init__(self, eog_ch, eog_len, seq_len, dim, drop_rate, num_heads, max_seq_len):
        super().__init__()
        self.conv_eog = ConvBlock(eog_ch, eog_len, seq_len, dim, drop_rate)
        self.ln_attn = nn.LayerNorm(dim)
        self.ln_final = nn.LayerNorm(dim)
        self.self_attn = Attention(dim=dim, n_heads=num_heads, max_seq_len=max_seq_len)

    def forward(self, eeg, eog):
        eog = self.conv_eog(eog.permute(0, 2, 1)).permute(0, 2, 1)
        attn_in = self.ln_attn(eog)
        eog = eog + self.self_attn(attn_in, attn_in, attn_in)
        eog_out = self.ln_final(eog)
        return eog_out.mean(dim=1)


class BackboneConcat(nn.Module):
    """EEG/EOG 直接拼接：无跨模态注意力、无门控，concat 后线性投影"""

    def __init__(self, eeg_ch, eog_ch, eeg_len, eog_len, seq_len, dim, drop_rate,
                 num_heads, max_seq_len):
        super().__init__()
        self.conv_eeg = ConvBlock(eeg_ch, eeg_len, seq_len, dim, drop_rate)
        self.conv_eog = ConvBlock(eog_ch, eog_len, seq_len, dim, drop_rate)
        self.concat_proj = nn.Linear(dim * 2, dim)
        self.ln_attn = nn.LayerNorm(dim)
        self.ln_final = nn.LayerNorm(dim)
        self.self_attn = Attention(dim=dim, n_heads=num_heads, max_seq_len=max_seq_len)

    def forward(self, eeg, eog):
        eeg = self.conv_eeg(eeg.permute(0, 2, 1)).permute(0, 2, 1)
        eog = self.conv_eog(eog.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.concat_proj(torch.cat([eeg, eog], dim=-1))
        attn_in = self.ln_attn(x)
        x = x + self.self_attn(attn_in, attn_in, attn_in)
        x_out = self.ln_final(x)
        return x_out.mean(dim=1)


class BackboneReversedAttn(Backbone):
    """反转注意力：EOG→EEG，无门控"""

    def forward(self, eeg, eog):
        eeg = self.conv_eeg(eeg.permute(0, 2, 1)).permute(0, 2, 1)
        eog = self.conv_eog(eog.permute(0, 2, 1)).permute(0, 2, 1)
        eeg_norm = self.ln_eeg(eeg)
        eog_norm = self.ln_eog(eog)
        cross_out = self.cross_attn(query=eog_norm, key=eeg_norm, value=eeg_norm)
        eog = eog + cross_out
        attn_in = self.ln_attn(eog)
        eog = eog + self.self_attn(attn_in, attn_in, attn_in)
        eog_out = self.ln_final(eog)
        return eog_out.mean(dim=1)


class BackboneReversedAttnGate(Backbone):
    """反转注意力：EOG→EEG，有门控"""

    def forward(self, eeg, eog):
        eeg = self.conv_eeg(eeg.permute(0, 2, 1)).permute(0, 2, 1)
        eog = self.conv_eog(eog.permute(0, 2, 1)).permute(0, 2, 1)
        alpha = self.gate_mlp(torch.cat([eeg, eog], dim=-1))
        eeg_norm = self.ln_eeg(eeg)
        eog_norm = self.ln_eog(eog)
        cross_out = self.cross_attn(query=eog_norm, key=eeg_norm, value=eeg_norm)
        eog = eog + alpha * cross_out
        attn_in = self.ln_attn(eog)
        eog = eog + self.self_attn(attn_in, attn_in, attn_in)
        eog_out = self.ln_final(eog)
        return eog_out.mean(dim=1)


class BackboneNoGate(Backbone):
    """移除动态门控：eeg + cross_out 直接融合，无 alpha 缩放"""

    def forward(self, eeg, eog):
        eeg = self.conv_eeg(eeg.permute(0, 2, 1)).permute(0, 2, 1)
        eog = self.conv_eog(eog.permute(0, 2, 1)).permute(0, 2, 1)
        eeg_norm = self.ln_eeg(eeg)
        eog_norm = self.ln_eog(eog)
        cross_out = self.cross_attn(query=eeg_norm, key=eog_norm, value=eog_norm)
        eeg = eeg + cross_out  # 无门控，直接融合
        attn_in = self.ln_attn(eeg)
        eeg = eeg + self.self_attn(attn_in, attn_in, attn_in)
        eeg_out = self.ln_final(eeg)
        return eeg_out.mean(dim=1)


class SingleStageNet(nn.Module):
    """backbone + reg_head"""

    def __init__(self, backbone, dim, drop_rate):
        super().__init__()
        self.backbone = backbone
        self.reg_head = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(dim // 2, dim // 4),
            nn.ReLU(),
            nn.Linear(dim // 4, 1),
        )

    def forward(self, eeg, eog):
        return self.reg_head(self.backbone(eeg, eog))
