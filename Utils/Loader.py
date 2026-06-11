import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset


# 每次仅读取当前受试者需计算的数据
class EEGDataset(Dataset):
    def __init__(self,
                 path: str,
                 subject_id,
                 eeg_ch,
                 eog_ch,
                 eeg_len,
                 eog_len):
        # 开启内存映射 mmap_mode='r'，仅需要时加载
        eeg_data = np.load(Path(path) / f'{subject_id}_eeg.npy', mmap_mode='r')
        eog_data = np.load(Path(path) / f'{subject_id}_eog.npy', mmap_mode='r')
        label_data = np.load(Path(path) / f'{subject_id}_label.npy', mmap_mode='r')

        self.len = len(label_data)
        # 重构形状 [len, seq_len, dim]
        self.eeg_data = eeg_data.reshape(-1, eeg_len, eeg_ch)
        self.eog_data = eog_data.reshape(-1, eog_len, eog_ch)
        self.label_data = label_data.reshape(-1, 1)

    def __getitem__(self, idx):
        # 数据类型转换为fp32
        eeg = torch.tensor(self.eeg_data[idx], dtype=torch.float32)
        eog = torch.tensor(self.eog_data[idx], dtype=torch.float32)
        label = torch.tensor(self.label_data[idx], dtype=torch.float32)

        return eeg, eog, label

    def __len__(self):
        return self.len


# 每次获取测试者id
def get_loaders(data_dir, test_sub_id, sub_num, val_num, batch_size, num_workers,
                eeg_ch, eog_ch, eeg_len, eog_len, seed=42,
                mode='train_val'):
    train_datasets = []
    val_datasets = []
    test_dataset = None

    # 除外层测试集外其余受试者 ID
    remaining_subs = [i for i in range(1, sub_num + 1) if i != test_sub_id]

    # train_val：optuna使用
    if mode == 'train_val':
        random.seed(seed)
        random.shuffle(remaining_subs)
        val_subs = remaining_subs[:val_num]
    # retrain：LOSO阶段使用
    elif mode == 'retrain' and val_num > 0:
        # LOSO 内部验证：从非测试受试者中固定划分
        random.seed(seed + test_sub_id)
        random.shuffle(remaining_subs)
        val_subs = remaining_subs[:val_num]
    else:
        val_subs = []

    # 根据划分好的ID索引构建 Dataset
    for sub_id in np.arange(1, sub_num + 1):
        dataset = EEGDataset(path=data_dir, subject_id=sub_id,
                             eeg_ch=eeg_ch, eog_ch=eog_ch,
                             eeg_len=eeg_len, eog_len=eog_len)
        if sub_id == test_sub_id:
            test_dataset = dataset
        elif sub_id in val_subs:
            val_datasets.append(dataset)
        else:
            train_datasets.append(dataset)

    # optuna将留出的测试者加入验证集，验证集变为 val_num+1 人
    if mode == 'train_val' and test_dataset is not None:
        val_datasets.append(test_dataset)

    # 拼接并打包为 DataLoader
    full_train_ds = ConcatDataset(train_datasets)
    train_loader = DataLoader(full_train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # LOSO模式
    if mode == 'retrain':
        if val_datasets:
            full_val_ds = ConcatDataset(val_datasets)
            val_loader = DataLoader(full_val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
            return train_loader, val_loader, test_loader
        return train_loader, None, test_loader
    # optuna模式
    else:
        full_val_ds = ConcatDataset(val_datasets)
        val_loader = DataLoader(full_val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        return train_loader, val_loader
