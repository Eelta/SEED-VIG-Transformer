"""
消融实验：6 组对照，LOSO 评估
"""
import torch
import torch.optim as optim
from torch import nn
import pandas as pd
import gc
import random
import numpy as np
from pathlib import Path
from Utils.Config import load_config
from Utils.Optuna import get_raw_model, EarlyStopping
from Utils.Loader import get_loaders
from Utils.Train_Val import train_epoch, val_loss, test_metrics
from Utils.Ablation_Models import (
    BackboneEEGOnly, BackboneEOGOnly, BackboneConcat,
    BackboneReversedAttn, BackboneReversedAttnGate,
    BackboneNoGate, SingleStageNet
)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _scalar(val):
    if isinstance(val, list):
        return (val[0] + val[-1]) / 2
    return val


def run_experiment(exp_name, make_net_fn, config):
    """LOSO"""
    metrics_list = []

    for sub_id in range(1, config.sub_num + 1):
        train_loader, val_loader, test_loader = get_loaders(
            data_dir='Data', test_sub_id=sub_id,
            sub_num=config.sub_num, val_num=config.val_num,
            batch_size=_scalar(config.batch_size),
            num_workers=config.num_workers,
            eeg_ch=config.eeg_ch, eog_ch=config.eog_ch,
            eeg_len=config.eeg_len, eog_len=config.eog_len,
            mode='loso')

        print(f'\n  [{exp_name}] LOSO {sub_id:>2}')

        net = make_net_fn(dim=_scalar(config.dim),
                          drop_rate=_scalar(config.drop_rate),
                          num_heads=_scalar(config.num_heads))
        net = net.to(DEVICE)
        torch._dynamo.config.recompile_limit = 256
        net = torch.compile(net)

        optimizer = optim.AdamW(net.parameters(),
                                lr=_scalar(config.lr),
                                weight_decay=_scalar(config.weight_decay))
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                         T_max=config.epoch_n)
        loss_fn = nn.HuberLoss(delta=_scalar(config.h_delta), reduction='none')
        log_prefix = f'  [{exp_name}] LOSO {sub_id:>2}'

        early_stopping = EarlyStopping(patience=config.patience)
        for epoch in range(config.epoch_n):
            avg_loss = train_epoch(loader=train_loader, net=net,
                                   optimizer=optimizer,
                                   device=DEVICE, loss_fn_per_sample=loss_fn)
            scheduler.step()
            val_l = val_loss(loader=val_loader, net=net, device=DEVICE,
                             loss_fn_per_sample=loss_fn)
            print(f'{log_prefix}, epoch {epoch + 1:>3}, '
                  f'train {avg_loss:.4f}, val {val_l:.4f}')

            early_stopping(val_l, net)
            if early_stopping.early_stop:
                print(f'{log_prefix}, early stopping at epoch {epoch + 1}')
                break

        get_raw_model(net).load_state_dict(early_stopping.best_model_weights)

        m = test_metrics(loader=test_loader, net=net, device=DEVICE)
        if m is not None:
            m['LOSO'] = sub_id
            metrics_list.append(m)
            print(f'  [{exp_name}] LOSO {sub_id:>2}  →  '
                  f'MAE {m["mae"]:.4f}, RMSE {m["rmse"]:.4f}, '
                  f'Pearson {m["pearson"]:.4f}, CCC {m["ccc"]:.4f}')

        del net, optimizer, train_loader, val_loader, test_loader
        gc.collect()
        torch.cuda.empty_cache()

    return metrics_list


def make_exp1(config):
    def fn(dim, drop_rate, num_heads):
        backbone = BackboneEEGOnly(
            eeg_ch=config.eeg_ch, eeg_len=config.eeg_len,
            seq_len=config.seq_len, dim=dim, drop_rate=drop_rate,
            num_heads=num_heads, max_seq_len=config.max_seq_len)
        return SingleStageNet(backbone, dim=dim, drop_rate=drop_rate)

    return fn


def make_exp2(config):
    def fn(dim, drop_rate, num_heads):
        backbone = BackboneEOGOnly(
            eog_ch=config.eog_ch, eog_len=config.eog_len,
            seq_len=config.seq_len, dim=dim, drop_rate=drop_rate,
            num_heads=num_heads, max_seq_len=config.max_seq_len)
        return SingleStageNet(backbone, dim=dim, drop_rate=drop_rate)

    return fn


def make_exp3(config):
    def fn(dim, drop_rate, num_heads):
        backbone = BackboneConcat(
            eeg_ch=config.eeg_ch, eog_ch=config.eog_ch,
            eeg_len=config.eeg_len, eog_len=config.eog_len,
            seq_len=config.seq_len, dim=dim, drop_rate=drop_rate,
            num_heads=num_heads, max_seq_len=config.max_seq_len)
        return SingleStageNet(backbone, dim=dim, drop_rate=drop_rate)

    return fn


def make_exp4(config):
    """反转注意力：EOG→EEG，无门控"""

    def fn(dim, drop_rate, num_heads):
        backbone = BackboneReversedAttn(
            eeg_ch=config.eeg_ch, eog_ch=config.eog_ch,
            eeg_len=config.eeg_len, eog_len=config.eog_len,
            seq_len=config.seq_len, dim=dim, drop_rate=drop_rate,
            num_heads=num_heads, max_seq_len=config.max_seq_len)
        return SingleStageNet(backbone, dim=dim, drop_rate=drop_rate)

    return fn


def make_exp5(config):
    """正向注意力：EEG→EOG，无门控"""

    def fn(dim, drop_rate, num_heads):
        backbone = BackboneNoGate(
            eeg_ch=config.eeg_ch, eog_ch=config.eog_ch,
            eeg_len=config.eeg_len, eog_len=config.eog_len,
            seq_len=config.seq_len, dim=dim, drop_rate=drop_rate,
            num_heads=num_heads, max_seq_len=config.max_seq_len)
        return SingleStageNet(backbone, dim=dim, drop_rate=drop_rate)

    return fn


def make_exp7(config):
    """反转注意力：EOG→EEG，有门控"""

    def fn(dim, drop_rate, num_heads):
        backbone = BackboneReversedAttnGate(
            eeg_ch=config.eeg_ch, eog_ch=config.eog_ch,
            eeg_len=config.eeg_len, eog_len=config.eog_len,
            seq_len=config.seq_len, dim=dim, drop_rate=drop_rate,
            num_heads=num_heads, max_seq_len=config.max_seq_len)
        return SingleStageNet(backbone, dim=dim, drop_rate=drop_rate)

    return fn


EXPERIMENTS = [
    ("1_EEG_Only", make_exp1),
    ("2_EOG_Only", make_exp2),
    ("3_Concat", make_exp3),
    ("4_Attn_Forward", make_exp5),
    ("5_Attn_Reversed", make_exp4),
    ("6_Attn_Reversed_Gate", make_exp7),
]

if __name__ == '__main__':
    config = load_config()

    # 打印实验列表并让用户选择
    print("\n  消融实验列表:")
    for name, _ in EXPERIMENTS:
        print(f"    {name}")
    user_input = input("\n  Enter experiment numbers (e.g. '1,3,5-7'), "
                       "or Enter for all: ").strip()

    selected = set()
    if not user_input:
        selected = set(range(len(EXPERIMENTS)))
    else:
        for part in user_input.split(','):
            part = part.strip()
            if '-' in part:
                a, b = part.split('-', 1)
                selected.update(range(int(a) - 1, int(b)))
            else:
                selected.add(int(part) - 1)

    Path("Results").mkdir(parents=True, exist_ok=True)
    all_results = []

    for idx in sorted(selected):
        if idx < 0 or idx >= len(EXPERIMENTS):
            continue
        exp_name, make_config_fn = EXPERIMENTS[idx]
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)

        print(f"\n{'=' * 60}")
        print(f"  Experiment: {exp_name}")
        print(f"{'=' * 60}")

        make_net_fn = make_config_fn(config)
        fold_metrics = run_experiment(exp_name, make_net_fn, config)

        if fold_metrics:
            df = pd.DataFrame(fold_metrics)
            mean_row = {k: v for k, v in df.mean(numeric_only=True).items()}
            mean_row['Experiment'] = exp_name
            mean_row['LOSO'] = 'Mean'
            all_results.append(mean_row)
            for _, row in df.iterrows():
                row_dict = row.to_dict()
                row_dict['Experiment'] = exp_name
                all_results.append(row_dict)

            print(f"\n  [{exp_name}] Mean → MAE {mean_row.get('mae', 0):.4f}, "
                  f"RMSE {mean_row.get('rmse', 0):.4f}, "
                  f"Pearson {mean_row.get('pearson', 0):.4f}, "
                  f"CCC {mean_row.get('ccc', 0):.4f}")

    results_df = pd.DataFrame(all_results)
    cols = ['Experiment', 'LOSO', 'mae', 'rmse', 'pearson', 'ccc']
    results_df = results_df[[c for c in cols if c in results_df.columns]]
    results_df.to_csv("Results/ablation.csv", index=False, encoding='utf-8')
    print(f"\n  Results saved to Results/ablation.csv")
