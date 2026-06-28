import torch
import torch.nn as nn
import torch.nn.functional as F


# 1. 生成旋转矩阵（实数版本，返回 cos 和 sin）
def precompute_freqs_cis(d: int, seq_len: int, theta: float = 10000.0):
    # 计算词向量元素两两分组之后，每组元素对应的旋转频率，选择偶数位置 f
    freqs = 1.0 / (theta ** (torch.arange(0, d, 2).float() / d))
    # 生成每个token位置索引 m = [0, 1,..., seq_len-1]
    m = torch.arange(seq_len, device=freqs.device)
    #  计算外积（所有位置的旋转角度矩阵m * f），[seq_len, d // 2]
    freqs = torch.outer(m, freqs).float()

    # 分别返回 cos(mf) 和 sin(mf)
    freqs_cos = freqs.cos()  # [seq_len, d // 2]
    freqs_sin = freqs.sin()  # [seq_len, d // 2]
    return freqs_cos, freqs_sin


# 2. 广播维度对齐函数
def reshape_for_broadcast(freqs: torch.Tensor, x: torch.Tensor):
    """
        x数据矩阵维度：[batch_size, seq_len, num_heads, head_dim]
        将 freqs [seq_len, head_dim // 2]
        重塑为适用于广播的形状 [1, seq_len, 1, head_dim // 2]
    """
    # 确保 x 至少有 2 个维度
    # 确保 freqs 的形状确实等于 [seq_len, head_dim]
    ndim = x.ndim
    assert ndim > 1, "x must have at least 2 dimensions"
    assert freqs.shape == (x.shape[1], x.shape[-1])

    # 动态构建形状，仅保留 seq_len 和 dim // 2 维度，其余设为 1
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs.view(*shape)


# 3. 旋转位置编码计算（实数版本，无复数运算）
def apply_rotary_emb(
        x: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
) -> torch.Tensor:
    # x.shape = [batch_size, seq_len, num_heads, head_dim]
    # 将 head_dim 的相邻两维配对：x_reshaped.shape = [..., head_dim//2, 2]
    x_ = x.float().reshape(*x.shape[:-1], -1, 2)

    # 分离配对元素：[..., head_dim//2]
    x0, x1 = x_[..., 0], x_[..., 1]

    # 广播对齐 cos/sin
    cos = reshape_for_broadcast(freqs_cos, x0)
    sin = reshape_for_broadcast(freqs_sin, x1)

    # 实数旋转公式：(x0 + i*x1) * (cos + i*sin) = (x0*cos - x1*sin) + i*(x0*sin + x1*cos)
    x0_out = x0 * cos - x1 * sin
    x1_out = x0 * sin + x1 * cos

    # 将配对维度重新组合，展平为 [..., head_dim]
    x_out = torch.stack([x0_out, x1_out], dim=-1).flatten(-2)

    return x_out.type_as(x)


# 4. Attention 模块
class Attention(nn.Module):
    def __init__(self, dim, n_heads, max_seq_len):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads  # RoPE 作用于 head_dim

        self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=False)

        # 预计算频率矩阵（基于 head_dim），safetensors 不支持 complex64，需分别存储 cos 和 sin
        freqs_cos, freqs_sin = precompute_freqs_cis(self.head_dim, max_seq_len * 2)
        self.register_buffer('freqs_cos', freqs_cos.float())
        self.register_buffer('freqs_sin', freqs_sin.float())

    # 从预计算的 cos/sin buffer 截取指定长度的 (cos, sin) 元组
    def _freqs(self, seq_len: int, device: torch.device):
        return (
            self.freqs_cos[:seq_len].to(device),
            self.freqs_sin[:seq_len].to(device))

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
        batch_size, q_seq_len, _ = query.shape
        _, k_seq_len, _ = key.shape

        # Q/K/V 投影分别作用于各自输入
        xq = self.wq(query)  # [B, qL, n_heads * head_dim]
        xk = self.wk(key)  # [B, kL, n_heads * head_dim]
        xv = self.wv(value)  # [B, kL, n_heads * head_dim]

        # shape: [batch_size, seq_len, num_heads, head_dim]
        xq = xq.view(batch_size, q_seq_len, self.n_heads, self.head_dim)
        xk = xk.view(batch_size, k_seq_len, self.n_heads, self.head_dim)
        xv = xv.view(batch_size, k_seq_len, self.n_heads, self.head_dim)

        # 截取当前 seqlen 所需的 cos/sin
        freqs_cos_q, freqs_sin_q = self._freqs(q_seq_len, xq.device)
        freqs_cos_k, freqs_sin_k = self._freqs(k_seq_len, xk.device)

        # attention 操作之前，应用旋转位置编码
        xq = apply_rotary_emb(xq, freqs_cos=freqs_cos_q, freqs_sin=freqs_sin_q)
        xk = apply_rotary_emb(xk, freqs_cos=freqs_cos_k, freqs_sin=freqs_sin_k)

        # 为进行矩阵乘法，需将 num_heads 移至前面 [batch_size, num_heads, seq_len, head_dim]
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # 使用 Flash Attention
        output = F.scaled_dot_product_attention(
            xq, xk, xv,
            is_causal=False,
            scale=None
        )  # [batch_size, num_heads, qL, head_dim]

        # 合并结果: [batch_size, qL, dim]
        output = output.transpose(1, 2).contiguous().view(batch_size, q_seq_len, -1)

        return self.wo(output)
