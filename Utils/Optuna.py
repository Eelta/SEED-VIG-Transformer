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
from .Train_Val import train_epoch, val_loss, test_metrics
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
               num_heads=model_params['num_heads'],
               max_seq_len=config.max_seq_len)


# 获取未编译的原始模型，兼容 torch.compile 的 _orig_mod 包装
def get_raw_model(net):
    return net._orig_mod if hasattr(net, '_orig_mod') else net


class EarlyStopping:
    def __init__(self, patience, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model_weights = None

    def __call__(self, val_loss, model):
        m = get_raw_model(model)
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model_weights = copy.deepcopy(m.state_dict())
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.best_model_weights = copy.deepcopy(m.state_dict())
            self.counter = 0


def train_loop(train_loader, val_loader, net, optimizer, scheduler,
               loss_fn, max_epochs,
               trial=None, log_prefix=''):
    early_stopping = EarlyStopping(patience=config.patience)
    epochs_run = 0

    for epoch in range(max_epochs):
        avg_loss = train_epoch(loader=train_loader, net=net, optimizer=optimizer,
                               device=DEVICE, loss_fn_per_sample=loss_fn)
        scheduler.step()
        val_l = val_loss(loader=val_loader, net=net, device=DEVICE,
                         loss_fn_per_sample=loss_fn)

        if log_prefix:
            print(f'{log_prefix}, S1 epoch {epoch + 1:>3}, '
                  f'train {avg_loss:.4f}, val {val_l:.4f}')

        if trial is not None:
            trial.report(val_l, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        early_stopping(val_l, net)
        epochs_run = epoch + 1

        if early_stopping.early_stop:
            if log_prefix:
                print(f'{log_prefix}, S1 early stopping at epoch {epoch + 1}')
            break

    get_raw_model(net).load_state_dict(early_stopping.best_model_weights)
    return early_stopping.best_loss, epochs_run


def objective(trial):
    # 超参数采样
    model_params = {
        'dim': trial.suggest_categorical('dim', config.dim),
        'drop_rate': trial.suggest_float('drop_rate', *config.drop_rate),
        'num_heads': trial.suggest_categorical('num_heads', config.num_heads),
    }
    batch_size = trial.suggest_categorical('batch_size', config.batch_size)

    h_delta = trial.suggest_float('h_delta', *config.h_delta)
    lr = trial.suggest_float('lr', *config.lr, log=True)
    weight_decay = trial.suggest_float('weight_decay', *config.weight_decay, log=True)

    loss_fn = nn.HuberLoss(delta=h_delta, reduction='none')

    data_path = os.path.join(_this_dir, '..', 'Data')
    train_loader, val_loader = get_loaders(
        data_dir=data_path,
        sub_num=config.sub_num, val_num=config.val_num, batch_size=batch_size,
        num_workers=config.num_workers,
        eeg_ch=config.eeg_ch, eog_ch=config.eog_ch,
        eeg_len=config.eeg_len, eog_len=config.eog_len,
        mode='optuna')

    log_prefix = f'[Trial {trial.number:>3}]'

    net = init_net(model_params)
    optimizer = optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epoch_n)

    net = net.to(DEVICE)
    torch._dynamo.config.recompile_limit = 256
    net = torch.compile(net)

    best_loss, epochs_run = train_loop(
        train_loader, val_loader, net, optimizer, scheduler,
        loss_fn, max_epochs=config.epoch_n,
        trial=trial, log_prefix=log_prefix)
    print(f'{log_prefix}, S1 best val loss {best_loss:.4f}')

    final_metrics = test_metrics(loader=val_loader, net=net, device=DEVICE)
    final_val_rmse = final_metrics['rmse'] if final_metrics is not None else float('inf')
    print(f'{log_prefix}, finished, val RMSE {final_val_rmse:.4f}')

    del net, optimizer, train_loader, val_loader
    gc.collect()
    torch.cuda.empty_cache()

    return final_val_rmse


def get_best_params(n_trials):
    random.seed(42)

    results_dir = os.path.join(_this_dir, '..', 'Results')
    os.makedirs(results_dir, exist_ok=True)
    best_params_path = os.path.join(_this_dir, '..', 'Configs', 'best_params.json')

    db_path = os.path.join(results_dir, 'optuna_study.db')
    storage_url = f'sqlite:///{db_path}'
    study_name = 'loso_optuna'

    pruner = optuna.pruners.MedianPruner()
    sampler = optuna.samplers.TPESampler(seed=42)

    study_exists = os.path.exists(db_path)
    study = optuna.create_study(
        direction='minimize', pruner=pruner, sampler=sampler,
        study_name=study_name, storage=storage_url,
        load_if_exists=True)

    existing_count = len(study.trials)
    remaining = n_trials - existing_count

    if study_exists:
        completed_before = len(study.get_trials(
            deepcopy=False, states=[optuna.trial.TrialState.COMPLETE]))
        print(f"  Resuming from existing study: {db_path}")
        print(f"  Existing: {existing_count} total, {completed_before} completed")
    else:
        print(f"  Hyperparameter Search (Optuna) {n_trials} trials")

    if remaining <= 0:
        print(f"  ✓ Already reached target ({existing_count} >= {n_trials}), skipping")
    else:
        print(f"  Need {remaining} more trials (target: {n_trials})")
        try:
            import shutil
            cache_dir = torch._inductor.codecache.base_cache_dir()
            if os.path.exists(cache_dir):
                shutil.rmtree(cache_dir, ignore_errors=True)
        except Exception:
            pass

        try:
            study.optimize(lambda trial: objective(trial),
                           n_trials=n_trials, gc_after_trial=True)
        except Exception as e:
            print(f"\n  ⚠ Optuna 优化中断: {e}")
            import traceback
            traceback.print_exc()
        else:
            print(f" Optuna Done")

    if study.trials:
        pruned_trials = study.get_trials(
            deepcopy=False, states=[optuna.trial.TrialState.PRUNED])
        completed_trials = study.get_trials(
            deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])
        best_trial = study.best_trial

        trials_info = {
            "param": ["optuna/Total_trials", "optuna/Completed_trials",
                      "optuna/Pruned_trials", "optuna/Best_trial_Val_RMSE"],
            "value": [len(study.trials), len(completed_trials),
                      len(pruned_trials), best_trial.value]
        }
        pd.DataFrame(trials_info).to_csv(
            os.path.join(results_dir, "optuna_trials_info.csv"),
            index=False, encoding='utf-8')

        best_dict = dict(best_trial.params)
        params_list = {"param": list(best_dict.keys()),
                       "value": list(best_dict.values())}
        pd.DataFrame(params_list).to_csv(
            os.path.join(results_dir, "optuna_best_params.csv"),
            index=False, encoding='utf-8')

        with open(best_params_path, 'w', encoding='utf-8') as f:
            json.dump(best_dict, f, indent=2, ensure_ascii=False)

        print(f"\n  Best params saved to: {os.path.abspath(best_params_path)}")
        print(f"  Results saved to: {os.path.abspath(results_dir)}")
    else:
        print("\n  ⚠ 无可用 trial 结果")

    print("=" * 50)
