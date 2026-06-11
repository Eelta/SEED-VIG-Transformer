import copy
import json
import os
import random
import pandas as pd
import optuna
import torch
import torch.optim as optim
import gc
from torch import nn
from .Train_Val import train_s1, train_s2, val_loss, test_metrics
from .LDS_FDS import get_lds_weights
from .Model import Net
from .Loader import get_loaders
from types import SimpleNamespace

# 降低 Optuna 自身日志冗余度
optuna.logging.set_verbosity(optuna.logging.WARNING)

# 读取配置文件
_this_dir = os.path.dirname(os.path.abspath(__file__))
_config_path = os.path.join(_this_dir, '..', 'Configs', 'config.json')
with open(_config_path, 'r', encoding='utf-8') as f:
    config_dict = json.load(f)
config = SimpleNamespace(**config_dict)

# 全局设备
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def init_net(model_params):
    return Net(eeg_ch=config.eeg_ch,
               eog_ch=config.eog_ch,
               eeg_len=config.eeg_len,
               eog_len=config.eog_len,
               seq_len=config.seq_len,
               dim=model_params['dim'],
               drop_rate=model_params['drop_rate'],
               p_mask=model_params['p_mask'],
               num_heads=model_params['num_heads'],
               max_seq_len=config.max_seq_len,
               num_bins=model_params['num_bins'], sigma=config.sigma)


# 早停类
class EarlyStopping:
    def __init__(self, patience, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model_weights = None

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model_weights = copy.deepcopy(model.state_dict())
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.best_model_weights = copy.deepcopy(model.state_dict())
            self.counter = 0


def objective(trial, sub_id):
    # 超参数采样
    model_params = {
        'dim': trial.suggest_categorical('dim', config.dim),
        'drop_rate': trial.suggest_float('drop_rate', *config.drop_rate),
        'p_mask': trial.suggest_float('p_mask', *config.p_mask),
        'num_heads': trial.suggest_categorical('num_heads', config.num_heads),
        'num_bins': trial.suggest_int('num_bins', *config.num_bins, step=10)
    }
    batch_size = trial.suggest_categorical('batch_size', config.batch_size)

    sigma = trial.suggest_float('sigma', *config.sigma)
    h_delta = trial.suggest_float('h_delta', *config.h_delta)
    lr_s1 = trial.suggest_float('lr_s1', *config.lr_s1, log=True)
    lr_s2 = trial.suggest_float('lr_s2', *config.lr_s2, log=True)
    weight_decay = trial.suggest_float('weight_decay', *config.weight_decay, log=True)

    # 损失函数定义
    loss_fn = nn.HuberLoss(delta=h_delta)
    loss_fn_per_sample = nn.HuberLoss(delta=h_delta, reduction='none')

    # 获取数据：18 训练 + 5 验证
    data_path = os.path.join(_this_dir, '..', 'Data')
    train_loader, val_loader = get_loaders(data_dir=data_path, test_sub_id=sub_id,
                                           sub_num=config.sub_num,
                                           val_num=config.val_num, batch_size=batch_size,
                                           num_workers=config.num_workers,
                                           eeg_ch=config.eeg_ch, eog_ch=config.eog_ch,
                                           eeg_len=config.eeg_len, eog_len=config.eog_len,
                                           mode='train_val')

    net = init_net(model_params)

    # 提前计算 LDS 权重和区间边缘
    lds_weights, bin_edges = get_lds_weights(train_loader, num_bins=model_params['num_bins'], sigma=sigma)

    # 优化器设置
    optimizer_s1 = optim.AdamW(net.parameters(), lr=lr_s1, weight_decay=weight_decay)
    optimizer_s2 = optim.AdamW(net.reg_head.parameters(), lr=lr_s2, weight_decay=weight_decay)
    scheduler_s1 = optim.lr_scheduler.CosineAnnealingLR(optimizer_s1, T_max=config.epoch_s1_n)
    scheduler_s2 = optim.lr_scheduler.CosineAnnealingLR(optimizer_s2, T_max=config.epoch_s2_n)

    # 模型移至设备
    net = net.to(DEVICE)
    lds_weights = lds_weights.to(DEVICE)
    bin_edges = bin_edges.to(DEVICE)

    # ─── Stage 1：训练 + 验证 + 早停 ───
    early_stopping = EarlyStopping(patience=15)
    best_s1_loss = float('inf')
    log_prefix = f'[Trial {trial.number:>3}]'

    for epoch in range(config.epoch_s1_n):
        train_s1(loader=train_loader, net=net, optimizer=optimizer_s1, epoch=epoch + 1,
                 sub_id=sub_id, device=DEVICE, loss_fn=loss_fn,
                 log_prefix=log_prefix)
        scheduler_s1.step()
        val_l = val_loss(loader=val_loader, net=net, device=DEVICE, loss_fn=loss_fn)

        val_metrics = test_metrics(loader=val_loader, net=net, device=DEVICE)
        val_r = val_metrics['rmse'] if val_metrics else float('inf')

        # 剪枝：基于 rmse
        trial.report(val_r, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        # 早停：基于损失
        early_stopping(val_l, net)

        if val_l is not None and val_l < best_s1_loss:
            best_s1_loss = val_l

        if early_stopping.early_stop:
            print(f'{log_prefix} S1 early stopping at epoch {epoch}')
            break

    # 恢复 Stage 1 最佳模型权重
    net.load_state_dict(early_stopping.best_model_weights)

    # 收集 Stage 1 特征供 FDS 使用
    net.eval()
    all_features, all_targets = [], []
    with torch.no_grad():
        for data in train_loader:
            eeg_feature, eog_feature, labels = [x.to(DEVICE) for x in data]
            feats = net.backbone(eeg=eeg_feature, eog=eog_feature)
            all_features.append(feats.cpu())
            all_targets.append(labels.cpu())
    cat_features = torch.cat(all_features, dim=0).to(DEVICE)
    cat_targets = torch.cat(all_targets, dim=0).to(DEVICE)
    net.fds.update_and_smooth_stats(cat_features, cat_targets, bin_edges)

    # ─── Stage 2：训练 + 每 epoch 验证 ───
    for param in net.backbone.parameters():
        param.requires_grad = False
    best_s2_loss = float('inf')

    for epoch in range(config.epoch_s2_n):
        train_s2(loader=train_loader, net=net, optimizer=optimizer_s2, bin_edges=bin_edges,
                 lds_weights=lds_weights, epoch=epoch + 1, sub_id=sub_id,
                 device=DEVICE, loss_fn_per_sample=loss_fn_per_sample,
                 log_prefix=log_prefix)
        scheduler_s2.step()
        val_l = val_loss(loader=val_loader, net=net, device=DEVICE, loss_fn=loss_fn)
        if val_l is not None and val_l < best_s2_loss:
            best_s2_loss = val_l

    # 最终评估
    final_metrics = test_metrics(loader=val_loader, net=net, device=DEVICE)
    if final_metrics is not None:
        final_val_rmse = final_metrics['rmse']
    else:
        final_val_rmse = float('inf')
    print(f'{log_prefix} finished, val RMSE {final_val_rmse:.4f}')

    del net, optimizer_s1, optimizer_s2, train_loader, val_loader
    gc.collect()
    torch.cuda.empty_cache()

    return final_val_rmse


def get_best_params(n_trials):
    """超参数搜索：保存到 best_params.json 和 Results/ 目录"""
    random.seed(42)
    rand_sub_id = random.randint(1, config.sub_num)
    print(f"  Hyperparameter Search (Optuna) {n_trials} trials")
    pruner = optuna.pruners.MedianPruner()
    study = optuna.create_study(direction='minimize', pruner=pruner)
    study.optimize(lambda trial: objective(trial, rand_sub_id), n_trials=n_trials)
    print(f" Optuna Done")

    pruned_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.PRUNED])
    best_trial = study.best_trial

    # ── 保存 Trials 汇总表 ──
    results_dir = os.path.join(_this_dir, '..', 'Results')
    os.makedirs(results_dir, exist_ok=True)

    trials_info = {
        "param": ["optuna/Total_trials", "optuna/Completed_trials",
                  "optuna/Pruned_trials", "optuna/Best_trial_Val_RMSE"],
        "value": [len(study.trials),
                  len(study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])),
                  len(pruned_trials),
                  best_trial.value]
    }
    pd.DataFrame(trials_info).to_csv(os.path.join(results_dir, "optuna_trials_info.csv"),
                                     index=False, encoding='utf-8')

    # ── 保存 Best Params 表 ──
    best_params_dict = {}
    best_params_list = {"param": [], "value": []}
    for key, value in best_trial.params.items():
        best_params_list["param"].append(key)
        best_params_list["value"].append(value)
        best_params_dict[key] = value
        setattr(config, key, value)

    pd.DataFrame(best_params_list).to_csv(os.path.join(results_dir, "optuna_best_params.csv"),
                                          index=False, encoding='utf-8')

    # 保存最佳超参数到 Configs 目录
    best_params_path = os.path.join(_this_dir, '..', 'Configs', 'best_params.json')
    with open(best_params_path, 'w', encoding='utf-8') as f:
        json.dump(best_params_dict, f, indent=2, ensure_ascii=False)
    print(f"\n  Best params saved to: {os.path.abspath(best_params_path)}")
    print(f"  Results saved to: {os.path.abspath(results_dir)}")
    print("=" * 50)
