import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple


# 1. 生成旋转矩阵
def precompute_freqs_cis(d: int, seq_len: int, theta: float = 10000.0):
    # 计算词向量元素两两分组之后，每组元素对应的旋转频率，选择偶数位置 f
    freqs = 1.0 / (theta ** (torch.arange(0, d, 2).float() / d))
    # 生成每个token位置索引 m = [0, 1,..., seq_len-1]
    m = torch.arange(seq_len, device=freqs.device)
    #  计算外积（所有位置的旋转角度矩阵m * f），[seq_len, d // 2]
    freqs = torch.outer(m, freqs).float()

    # 转为复数向量，cos(mf) + i*sin(mf)，模长i=1
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


# 2. 广播维度对齐函数
def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """
        x数据矩阵维度：[batch_size, seq_len, num_heads, head_dim]
        将 freqs_cis [seq_len, head_dim // 2]
        重塑为适用于广播的形状 [1, seq_len, 1, head_dim // 2]
    """
    # 确保 x 至少有 2 个维度
    # 确保 freqs_cis 的形状确实等于 [seq_len, head_dim]
    ndim = x.ndim
    assert ndim > 1, "x must have at least 2 dimensions"
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])

    # 动态构建形状，仅保留 seq_len 和 dim // 2 维度，其余设为 1
    # shape = freqs_cis.unsqueeze(0).unsqueeze(2).shape
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


# 3. 旋转位置编码计算
def apply_rotary_emb(
        xq: torch.Tensor,
        xk: torch.Tensor,
        freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # xq.shape = [batch_size, seq_len, num_heads, head_dim]
    # xq_.shape = [batch_size, seq_len, num_heads, head_dim // 2, 2]
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 2)

    # 转为复数域 [batch_size, seq_len, num_heads, head_dim // 2]
    xq_ = torch.view_as_complex(xq_)
    xk_ = torch.view_as_complex(xk_)

    # 广播对齐 freqs_cis
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)

    # 1.旋转操作 2.将结果转回实数域，并展平最后两维
    # 使用 flatten(-2) 会将 [..., head_dim//2, 2] 展平为 [..., head_dim]
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(-2)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(-2)

    return xq_out.type_as(xq), xk_out.type_as(xk)


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

        # 预计算频率矩阵（基于 head_dim），拆为实部/虚部两个 buffer
        # safetensors 不支持 complex64，需分别存储 cos 和 sin
        freqs_cis = precompute_freqs_cis(self.head_dim, max_seq_len * 2)
        self.register_buffer('freqs_cos', freqs_cis.real.float())
        self.register_buffer('freqs_sin', freqs_cis.imag.float())

    def forward(self, x: torch.Tensor):
        batch_size, seq_len, _ = x.shape  # batch_size, seq_len, dim
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        # shape: [batch_size, seq_len, num_heads, head_dim]
        xq = xq.view(batch_size, seq_len, self.n_heads, self.head_dim).float()
        xk = xk.view(batch_size, seq_len, self.n_heads, self.head_dim).float()
        xv = xv.view(batch_size, seq_len, self.n_heads, self.head_dim).float()

        # 只截取当前 seqlen 所需的旋转矩阵，由 cos/sin buffer 重构复数张量
        freqs_cis = torch.complex(
            self.freqs_cos[:seq_len].to(xq.device),
            self.freqs_sin[:seq_len].to(xq.device))

        # attention 操作之前，应用旋转位置编码
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        # 为进行矩阵乘法，需将 num_heads 移至前面
        # xq.shape = [batch_size, num_heads, seq_len, head_dim]
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # 计算 Attention Scores
        # xk 矩阵转置：[batch_size, num_heads, head_dim, seq_len]
        scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)  # 点积
        scores = F.softmax(scores, dim=-1).type_as(xq)

        # 乘以 Value
        output = torch.matmul(scores, xv)  # [batch_size, num_heads, seq_len, head_dim]

        # 合并结果: [batch_size, seq_len, dim]
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

        return self.wo(output)
