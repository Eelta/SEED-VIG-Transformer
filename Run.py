import pandas as pd
import torch
from torch import nn
import torch.optim as optim
from Utils.Train_Val import train_s1, train_s2, val_loss, test_metrics
from Utils.LDS_FDS import get_lds_weights
from Utils.Model import Net
from Utils.Loader import get_loaders
from Utils.Config import load_config
from safetensors.torch import save_file, load_file
import gc
import copy
import os
import random
import numpy as np

# 全局设备
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def test(loader, sub_id):
    net = init_net()
    net.load_state_dict(load_file(f"Models/LOSO{sub_id}_best_model.safetensors"))
    net.to(DEVICE)
    return test_metrics(loader=loader, net=net, device=DEVICE)


def init_net():
    net = Net(eeg_ch=config.eeg_ch,
              eog_ch=config.eog_ch,
              eeg_len=config.eeg_len,
              eog_len=config.eog_len,
              seq_len=config.seq_len,
              dim=config.dim,
              drop_rate=config.drop_rate,
              p_mask=config.p_mask,
              num_heads=config.num_heads,
              max_seq_len=config.max_seq_len,
              num_bins=config.num_bins, sigma=config.sigma)
    return net


if __name__ == '__main__':
    # 确认是否进行超参数优化
    user_input = input(
        "Run hyperparameter optimization? (y/n, default n): ").strip().lower()
    do_optuna = user_input in ('y', 'yes')

    # 超参数搜索
    if do_optuna:
        from Utils.Optuna import get_best_params

        config_optuna = load_config()
        get_best_params(config_optuna.n_trials)

    config = load_config()

    # 确认是否进行LOSO
    print("Optuna done, start LOSO?")
    user_input = input("(y/n, default y): ").strip().lower()
    do_loso = user_input not in ('n', 'no')

    if not do_loso:
        print("Done")
        import sys

        sys.exit(0)

    loss_fn = nn.HuberLoss(delta=config.h_delta)
    loss_fn_per_sample = nn.HuberLoss(delta=config.h_delta, reduction='none')

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    metrics_list = []
    os.makedirs("Models", exist_ok=True)
    os.makedirs("Results", exist_ok=True)

    print(f"\n{'=' * 50}")
    print(f"  LOSO Evaluation ({config.sub_num} subjects)")
    print(f"  S1 max {config.epoch_s1_n} ep + S2 max {config.epoch_s2_n} ep")
    print(f"{'=' * 50}\n")

    # LOSO：每 fold 训练+验证+测试
    for sub_id in range(1, config.sub_num + 1):
        train_loader, val_loader, test_loader = get_loaders(
            data_dir='Data', test_sub_id=sub_id,
            sub_num=config.sub_num, val_num=config.val_num,
            batch_size=config.batch_size, num_workers=config.num_workers,
            eeg_ch=config.eeg_ch, eog_ch=config.eog_ch,
            eeg_len=config.eeg_len, eog_len=config.eog_len,
            mode='retrain')

        print(f'\n  LOSO {sub_id:>2}: train {config.sub_num - 1 - config.val_num} sub, '
              f'val {config.val_num} sub, test 1 sub')

        net = init_net()
        lds_weights, bin_edges = get_lds_weights(train_loader,
                                                 num_bins=config.num_bins,
                                                 sigma=config.sigma)

        optimizer_s1 = optim.AdamW(net.parameters(), lr=config.lr_s1,
                                   weight_decay=config.weight_decay)
        scheduler_s1 = optim.lr_scheduler.CosineAnnealingLR(optimizer_s1, T_max=config.epoch_s1_n)
        optimizer_s2 = optim.AdamW(net.reg_head.parameters(), lr=config.lr_s2,
                                   weight_decay=config.weight_decay)
        scheduler_s2 = optim.lr_scheduler.CosineAnnealingLR(optimizer_s2, T_max=config.epoch_s2_n)

        # 模型移至设备
        net = net.to(DEVICE)
        lds_weights = lds_weights.to(DEVICE)
        bin_edges = bin_edges.to(DEVICE)

        # Stage 1：训练 + 验证 + 早停 + 追踪最优
        patience_s1 = 15
        no_improve_s1 = 0
        best_s1_val = float('inf')
        best_s1_state = None

        # 设置coral_weight与target_loader，开启CORAL 协方差对齐损失
        for epoch in range(config.epoch_s1_n):
            train_s1(loader=train_loader, net=net, optimizer=optimizer_s1,
                     epoch=epoch + 1, sub_id=sub_id,
                     device=DEVICE, loss_fn=loss_fn,
                     target_loader=test_loader, coral_weight=config.coral_weight)
            scheduler_s1.step()

            val_l = val_loss(loader=val_loader, net=net, device=DEVICE, loss_fn=loss_fn)
            if val_l is not None and val_l < best_s1_val:
                best_s1_val = val_l
                no_improve_s1 = 0
                best_s1_state = copy.deepcopy(net.state_dict())
            elif val_l is not None:
                no_improve_s1 += 1

            if no_improve_s1 >= patience_s1:
                print(f'  LOSO {sub_id:>2}, S1 early stop at epoch {epoch + 1}')
                break

        # 恢复 Stage 1 最优模型
        net.load_state_dict(best_s1_state)
        print(f'  LOSO {sub_id:>2}, S1 best val loss {best_s1_val:.4f}')

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

        # Stage 2：冻结 backbone，训练 reg_head + FDS + LDS
        for param in net.backbone.parameters():
            param.requires_grad = False
        patience_s2 = 10
        no_improve_s2 = 0
        best_s2_val = float('inf')
        best_s2_state = None

        # 设置bin_edges和lds_weights，开启LDS与FDS
        for epoch in range(config.epoch_s2_n):
            train_s2(loader=train_loader, net=net, optimizer=optimizer_s2,
                     bin_edges=bin_edges, lds_weights=lds_weights,
                     epoch=epoch + 1, sub_id=sub_id,
                     device=DEVICE, loss_fn_per_sample=loss_fn_per_sample)
            scheduler_s2.step()

            val_l = val_loss(loader=val_loader, net=net, device=DEVICE, loss_fn=loss_fn)
            if val_l is not None and val_l < best_s2_val:
                best_s2_val = val_l
                no_improve_s2 = 0
                best_s2_state = copy.deepcopy(net.state_dict())
            elif val_l is not None:
                no_improve_s2 += 1

            if no_improve_s2 >= patience_s2:
                print(f'  LOSO {sub_id:>2}, S2 early stop at epoch {epoch + 1}')
                break

        # 恢复最优 S2 模型并保存为最终模型
        net.load_state_dict(best_s2_state)
        save_file(net.state_dict(), f"Models/LOSO{sub_id}_best_model.safetensors")
        print(f'  LOSO {sub_id:>2}, model saved\n')

        del net, optimizer_s1, optimizer_s2, train_loader, val_loader
        gc.collect()
        torch.cuda.empty_cache()

        # 测试
        loso_metrics = test(loader=test_loader, sub_id=sub_id)
        if loso_metrics is not None:
            metrics_list.append({
                "LOSO": sub_id,
                "MAE": loso_metrics["mae"],
                "RMSE": loso_metrics["rmse"],
                "Pearson": loso_metrics["pearson"],
                "CCC": loso_metrics["ccc"],
            })
            print(
                f'LOSO {sub_id:>2}, test  →  MAE {loso_metrics["mae"]:.4f}, '
                f'RMSE {loso_metrics["rmse"]:.4f}, '
                f'Pearson {loso_metrics["pearson"]:.4f}, CCC {loso_metrics["ccc"]:.4f}\n')

    loso_df = pd.DataFrame(metrics_list)
    loso_df.to_csv("Results/loso_per_subject.csv", index=False, encoding='utf-8')

    mean_metrics = loso_df[["MAE", "RMSE", "Pearson", "CCC"]].mean().to_dict()
    print(f"{'=' * 50}")
    print(f"  LOSO Mean Results ({config.sub_num} subjects)")
    print(f"{'=' * 50}")
    for k, v in mean_metrics.items():
        print(f"  {k:>8}: {v:.4f}")
    print(f"{'=' * 50}")

    # 保存均值到 CSV
    mean_df = pd.DataFrame([mean_metrics])
    mean_df.to_csv("Results/loso_mean.csv", index=False, encoding='utf-8')
    print(f"\n  Results saved to Results/")
