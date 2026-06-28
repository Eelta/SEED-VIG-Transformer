import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset


class EEGDataset(Dataset):
    def __init__(self, path: str, subject_id, eeg_ch, eog_ch, eeg_len, eog_len):
        eeg_data = np.load(Path(path) / f'{subject_id}_eeg.npy', mmap_mode='r')
        eog_data = np.load(Path(path) / f'{subject_id}_eog.npy', mmap_mode='r')
        label_data = np.load(Path(path) / f'{subject_id}_label.npy', mmap_mode='r')

        self.len = len(label_data)
        self.eeg_data = eeg_data.reshape(-1, eeg_len, eeg_ch)
        self.eog_data = eog_data.reshape(-1, eog_len, eog_ch)
        self.label_data = label_data.reshape(-1, 1)

    def __getitem__(self, idx):
        eeg = torch.tensor(self.eeg_data[idx], dtype=torch.float32)
        eog = torch.tensor(self.eog_data[idx], dtype=torch.float32)
        label = torch.tensor(self.label_data[idx], dtype=torch.float32)
        return eeg, eog, label

    def __len__(self):
        return self.len


def get_loaders(data_dir, sub_num, val_num, batch_size, num_workers,
                eeg_ch, eog_ch, eeg_len, eog_len, seed=42,
                test_sub_id=None, mode=None):
    if mode not in ('optuna', 'loso'):
        raise ValueError("mode must be 'optuna' or 'loso'")

    datasets = {}
    for sub_id in range(1, sub_num + 1):
        datasets[sub_id] = EEGDataset(path=data_dir, subject_id=sub_id,
                                      eeg_ch=eeg_ch, eog_ch=eog_ch,
                                      eeg_len=eeg_len, eog_len=eog_len)

    if mode == 'optuna':
        rng = random.Random(seed)
        all_subs = list(range(1, sub_num + 1))
        rng.shuffle(all_subs)
        val_subs = all_subs[:val_num]
        train_subs = all_subs[val_num:]

        train_ds = ConcatDataset([datasets[s] for s in train_subs])
        val_ds = ConcatDataset([datasets[s] for s in val_subs])
        train_ldr = DataLoader(train_ds, batch_size=batch_size,
                               shuffle=True, num_workers=num_workers)
        val_ldr = DataLoader(val_ds, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers)
        return train_ldr, val_ldr

    # mode == 'loso'
    test_ds = datasets[test_sub_id]
    remaining = [i for i in range(1, sub_num + 1) if i != test_sub_id]
    rng = random.Random(seed + test_sub_id)
    rng.shuffle(remaining)
    val_subs = remaining[:val_num] if val_num > 0 else []
    train_subs = remaining[val_num:] if val_num > 0 else remaining

    train_ds = ConcatDataset([datasets[s] for s in train_subs])
    train_ldr = DataLoader(train_ds, batch_size=batch_size,
                           shuffle=True, num_workers=num_workers)
    val_ldr = None
    if val_subs:
        val_ds = ConcatDataset([datasets[s] for s in val_subs])
        val_ldr = DataLoader(val_ds, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers)
    test_ldr = DataLoader(test_ds, batch_size=batch_size,
                          shuffle=False, num_workers=num_workers)
    return train_ldr, val_ldr, test_ldr
