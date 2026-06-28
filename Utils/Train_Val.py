import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
import scipy.stats


# 训练
def train_epoch(loader, net, optimizer, device, loss_fn_per_sample):
    net.train()
    total_loss = 0.0
    n_batches = 0

    for i, data in enumerate(loader, 1):
        optimizer.zero_grad()
        eeg_feature, eog_feature, labels = [x.to(device) for x in data]
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            pre = net(eeg_feature, eog_feature)
            base_loss = loss_fn_per_sample(pre.view(-1), labels.view(-1))
        loss = base_loss.mean()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


# 验证损失
def val_loss(loader, net, device, loss_fn_per_sample):
    with torch.no_grad():
        net.eval()
        total_loss = 0
        total_samples = 0
        for i, data in enumerate(loader, 1):
            eeg_feature, eog_feature, labels = [x.to(device) for x in data]
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                pre = net(eeg_feature, eog_feature)
                loss = loss_fn_per_sample(pre.view(-1), labels.view(-1))
            total_loss += loss.sum().item()
            total_samples += labels.size(0)
    return total_loss / total_samples


# 测试指标
def test_metrics(loader, net, device):
    with torch.no_grad():
        net.eval()
        predictions = []
        true_labels = []
        for i, data in enumerate(loader, 1):
            eeg_feature, eog_feature, labels = [x.to(device) for x in data]
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                pre = net(eeg_feature, eog_feature)
            predictions.append(pre.detach().float().cpu().numpy())
            true_labels.append(labels.detach().float().cpu().numpy())

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
