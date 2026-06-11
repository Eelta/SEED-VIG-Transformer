import numpy as np
import torch
from .LDS_FDS import get_bin_idx
from sklearn.metrics import mean_absolute_error, mean_squared_error
import scipy.stats


# CORAL 协方差对齐损失
def _coral_loss(source, target):
    """计算源域和目标域特征的协方差矩阵之间的 Frobenius 距离"""
    source = source.float()
    target = target.float()
    src_cov = torch.cov(source.T)
    tgt_cov = torch.cov(target.T)
    return ((src_cov - tgt_cov) ** 2).sum()  # pure Frobenius norm², ~1~100 量级


# 1阶段训练
def train_s1(loader, net, optimizer, epoch, sub_id, device, loss_fn, coral_weight,
             log_prefix='LOSO', target_loader=None):
    net.train()
    total_loss = 0.0
    total_huber = 0.0
    total_coral = 0.0
    n_batches = 0

    target_iter = None
    if target_loader is not None:
        # 测试集数量远低于训练集，将其转为迭代器
        target_iter = iter(target_loader)

    for i, data in enumerate(loader, 1):
        optimizer.zero_grad()
        eeg_feature, eog_feature, labels = [x.to(device) for x in data]

        # 提取 backbone 特征一次，避免 CORAL 时重复计算
        src_feat = net.backbone(eeg_feature, eog_feature)  # [B, dim]
        pre = net.reg_head(net.fds(src_feat))  # FDS passthrough（train_s1 无 targets）
        huber_loss = loss_fn(pre, labels)

        if target_iter is not None and coral_weight > 0:
            try:
                tgt_eeg, tgt_eog, _ = next(target_iter)
            except StopIteration:
                # 如目标域抽完（触发 StopIteration 异常）
                # 重新初始化迭代器，继续无限循环抽取
                target_iter = iter(target_loader)
                tgt_eeg, tgt_eog, _ = next(target_iter)
            tgt_eeg, tgt_eog = tgt_eeg.to(device), tgt_eog.to(device)
            tgt_feat = net.backbone(tgt_eeg, tgt_eog)
            coral = _coral_loss(src_feat, tgt_feat)
            loss = huber_loss + coral_weight * coral
            total_coral += coral.item()
        else:
            loss = huber_loss

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_huber += huber_loss.item()
        n_batches += 1

    avg_loss = total_loss / n_batches
    avg_huber = total_huber / n_batches
    avg_coral = total_coral / n_batches if total_coral > 0 else 0.0

    if log_prefix == 'LOSO':
        print(
            f'{log_prefix} {sub_id:>2}, Train Stage 1, epoch {epoch:>3}, '
            f'avg loss {avg_loss:.4f} (H={avg_huber:.4f} C={avg_coral:.4f})')
    else:
        print(
            f'{log_prefix}, Train Stage 1, epoch {epoch:>3}, '
            f'avg loss {avg_loss:.4f} (H={avg_huber:.4f} C={avg_coral:.4f})')


# 2阶段训练
def train_s2(loader, net, optimizer, bin_edges, lds_weights, epoch, sub_id,
             device, loss_fn_per_sample, log_prefix='LOSO'):
    # 仅 reg_head 和 FDS 需训练模式；backbone 已冻结，保持 eval 避免 Dropout/EOG 随机丢弃
    net.train()
    net.backbone.eval()

    total_loss = 0.0
    n_batches = 0
    total_batches = len(loader)
    # 每个 epoch 内前 80% batch α=1.0（校准特征），后 20% 线性衰减到 0.0（原始特征）
    threshold = max(1, int(total_batches * 0.8))

    for i, data in enumerate(loader, 1):
        optimizer.zero_grad()
        # Per-batch alpha 衰减
        if i <= threshold:
            alpha = 1.0
        else:
            alpha = 1.0 - (i - threshold) / (total_batches - threshold)
        net.fds.alpha = alpha

        eeg_feature, eog_feature, labels = [x.to(device) for x in data]
        # 传入targets和bin_edges，执行FDS
        pre = net(eeg_feature, eog_feature, targets=labels, bin_edges=bin_edges)
        base_loss = loss_fn_per_sample(pre, labels)
        # 根据目标值查找对应的 LDS 权重
        batch_bins = get_bin_idx(labels, bin_edges)
        batch_weights = lds_weights[batch_bins]
        # 应用 LDS 权重并求平均
        loss = (base_loss * batch_weights).mean()

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    avg_loss = total_loss / n_batches
    if log_prefix == 'LOSO':
        print(
            f'{log_prefix} {sub_id:>2}, Train Stage 2, epoch {epoch:>3}, '
            f'avg loss {avg_loss:.4f}')
    else:
        print(
            f'{log_prefix}, Train Stage 2, epoch {epoch:>3}, '
            f'avg loss {avg_loss:.4f}')


# 验证损失
def val_loss(loader, net, device, loss_fn):
    with torch.no_grad():
        net.eval()
        total_loss = 0
        for i, data in enumerate(loader, 1):
            eeg_feature, eog_feature, labels = [x.to(device) for x in data]
            pre = net(eeg_feature, eog_feature)
            loss = loss_fn(pre, labels)
            total_loss += loss.mean().item()
    return total_loss / len(loader)


# 测试指标
def test_metrics(loader, net, device):
    with torch.no_grad():
        net.eval()
        predictions = []
        true_labels = []
        for i, data in enumerate(loader, 1):
            eeg_feature, eog_feature, labels = [x.to(device) for x in data]
            pre = net(eeg_feature, eog_feature)
            predictions.append(pre.detach().cpu().numpy())
            true_labels.append(labels.detach().cpu().numpy())

    preds = np.concatenate(predictions, axis=0).reshape(-1)
    refs = np.concatenate(true_labels, axis=0).reshape(-1)
    mae = mean_absolute_error(refs, preds)
    rmse = np.sqrt(mean_squared_error(refs, preds))
    pred_mean, ref_mean = np.mean(preds), np.mean(refs)
    pred_var, ref_var = np.var(preds), np.var(refs)
    if pred_var < 1e-8 or ref_var < 1e-8:
        print(f'[DEBUG] test_metrics: pred_var={pred_var:.4e}, ref_var={ref_var:.4e}, '
              f'pred_mean={pred_mean:.4f}, ref_mean={ref_mean:.4f}, '
              f'pred_min={preds.min():.4f}, pred_max={preds.max():.4f}')
        pearson = 0.0
        ccc = 0.0
    else:
        pearson, _ = scipy.stats.pearsonr(preds, refs)
        covariance = np.cov(preds, refs, ddof=0)[0][1]
        ccc = (2 * covariance) / (pred_var + ref_var + (pred_mean - ref_mean) ** 2)
    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "pearson": float(pearson),
        "ccc": float(ccc)
    }
