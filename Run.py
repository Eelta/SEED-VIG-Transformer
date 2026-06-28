import pandas as pd
import torch
from torch import nn
import torch.optim as optim
from Utils.Train_Val import train_epoch, val_loss, test_metrics
from Utils.Model import Net
from Utils.Loader import get_loaders
from Utils.Config import load_config
from Utils.Optuna import get_raw_model, EarlyStopping
from safetensors.torch import save_file, load_file
import gc
from pathlib import Path
import random
import numpy as np

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def test(loader, sub_id):
    net = init_net()
    net.load_state_dict(load_file(f"Models/LOSO{sub_id}_best_model.safetensors"))
    net.to(DEVICE)
    net = torch.compile(net)
    return test_metrics(loader=loader, net=net, device=DEVICE)


def init_net():
    return Net(eeg_ch=config.eeg_ch, eog_ch=config.eog_ch,
               eeg_len=config.eeg_len, eog_len=config.eog_len,
               seq_len=config.seq_len, dim=config.dim,
               drop_rate=config.drop_rate, num_heads=config.num_heads,
               max_seq_len=config.max_seq_len)


if __name__ == '__main__':
    user_input = input("Run hyperparameter optimization? (y/n, default n): ").strip().lower()
    do_optuna = user_input in ('y', 'yes')

    if do_optuna:
        from Utils.Optuna import get_best_params

        config_optuna = load_config()
        get_best_params(config_optuna.n_trials)

    config = load_config()

    print("Optuna done, start LOSO")

    loss_fn = nn.HuberLoss(delta=config.h_delta, reduction='none')

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    metrics_list = []
    Path("Models").mkdir(parents=True, exist_ok=True)
    Path("Results").mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 50}")
    print(f"  LOSO Evaluation ({config.sub_num} subjects)")
    print(f"  Max {config.epoch_n} epochs")
    print(f"{'=' * 50}\n")

    for sub_id in range(1, config.sub_num + 1):
        train_loader, val_loader, test_loader = get_loaders(
            data_dir='Data', test_sub_id=sub_id,
            sub_num=config.sub_num, val_num=config.val_num,
            batch_size=config.batch_size, num_workers=config.num_workers,
            eeg_ch=config.eeg_ch, eog_ch=config.eog_ch,
            eeg_len=config.eeg_len, eog_len=config.eog_len,
            mode='loso')

        print(f'\n  LOSO {sub_id:>2}: train {config.sub_num - 1 - config.val_num} sub, '
              f'val {config.val_num} sub, test 1 sub')

        net = init_net()
        net = net.to(DEVICE)
        torch._dynamo.config.recompile_limit = 256
        net = torch.compile(net)

        optimizer = optim.AdamW(net.parameters(), lr=config.lr,
                                weight_decay=config.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                         T_max=config.epoch_n)
        log_prefix = f'  LOSO {sub_id:>2}'

        early_stopping = EarlyStopping(patience=config.patience)
        for epoch in range(config.epoch_n):
            avg_loss = train_epoch(loader=train_loader, net=net, optimizer=optimizer,
                                   device=DEVICE, loss_fn_per_sample=loss_fn)
            scheduler.step()
            val_l = val_loss(loader=val_loader, net=net, device=DEVICE,
                             loss_fn_per_sample=loss_fn)
            if log_prefix:
                print(f'{log_prefix}, epoch {epoch + 1:>3}, '
                      f'train {avg_loss:.4f}, val {val_l:.4f}')
            early_stopping(val_l, net)
            if early_stopping.early_stop:
                print(f'{log_prefix}, early stopping at epoch {epoch + 1}')
                break

        get_raw_model(net).load_state_dict(early_stopping.best_model_weights)
        save_file(get_raw_model(net).state_dict(),
                  f"Models/LOSO{sub_id}_best_model.safetensors")
        print(f'  LOSO {sub_id:>2}, model saved\n')

        del net, optimizer, train_loader, val_loader
        gc.collect()
        torch.cuda.empty_cache()

        m = test(loader=test_loader, sub_id=sub_id)
        if m is not None:
            metrics_list.append({
                "LOSO": sub_id, "MAE": m["mae"], "RMSE": m["rmse"],
                "Pearson": m["pearson"], "CCC": m["ccc"],
            })
            print(f'LOSO {sub_id:>2}, test  →  MAE {m["mae"]:.4f}, '
                  f'RMSE {m["rmse"]:.4f}, '
                  f'Pearson {m["pearson"]:.4f}, CCC {m["ccc"]:.4f}\n')

    loso_df = pd.DataFrame(metrics_list)
    loso_df.to_csv("Results/loso_per_subject.csv", index=False, encoding='utf-8')

    mean_metrics = loso_df[["MAE", "RMSE", "Pearson", "CCC"]].mean().to_dict()
    print(f"{'=' * 50}")
    print(f"  LOSO Mean Results ({config.sub_num} subjects)")
    print(f"{'=' * 50}")
    for k, v in mean_metrics.items():
        print(f"  {k:>8}: {v:.4f}")
    print(f"{'=' * 50}")

    mean_df = pd.DataFrame([mean_metrics])
    mean_df.to_csv("Results/loso_mean.csv", index=False, encoding='utf-8')
    print(f"\n  Results saved to Results/")
