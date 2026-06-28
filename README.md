[中文](./README_CN.md) | **English**

## Description

End-to-end Transformer training on the SEED-VIG dataset for vigilance (PERCLOS) regression prediction under strict LOSO (Leave-One-Subject-Out) cross-validation.

## Usage

1. Download the dataset from: https://huggingface.co/datasets/Curryjiang/SEED-VIG

2. Place raw `.mat` EEG data in `Raw_Data/mat_data` and labels in `Raw_Data/perclos_labels`

3. Run `Raw_Data/p_process.py` for 7-channel forehead EEG/EOG extraction and preprocessing

4. Run `Run.py` for hyperparameter optimization, training, validation, and LOSO evaluation

Configuration parameters can be adjusted in `Configs/config.json`.

## Model

1. Depthwise separable convolution (EEG/EOG)
2. Dynamic gate — per-timestep adaptive fusion weight
3. Cross-modal multi-head attention (EEG queries EOG)
4. Self-attention with RoPE encoding
5. Regression head (MLP)

## Pipeline

1. Split 23 subjects into training + validation sets; perform hyperparameter search with Optuna (100 trials).

2. LOSO evaluation: 22 subjects for training (with 4 held out for validation), 1 subject for testing — repeated for all 23 subjects.

## Data

- `eeg.npy` shape: [885000, 7] — 7 forehead channels
- `eog.npy` shape: [885000, 2] — VEOf + HEOf
- `label.npy` shape: [885000, 1] — PERCLOS labels

All 7 forehead electrode channels are used. The spatial-weight method locks VEO on vertically-aligned electrodes and HEO on horizontally-aligned electrodes (bipolar pattern). An adaptive Pearson correlation threshold identifies and removes EOG artifacts from EEG independent components. Cleaned EEG and EOG signals undergo band-pass filtering, 5σ clipping, and Z-score normalization.

## Results

| **Metric** | **Mean** | **SD** | **SEM** | **95% CI Low** | **95% CI High** | **Min** | **Max** |
| ---------- | -------- | ------ | ------- | -------------- | --------------- | ------- | ------- |
| **MAE**    | 0.1315   | 0.0445 | 0.0093  | 0.1122         | 0.1507          | 0.0647  | 0.2283  |
| **RMSE**   | 0.1655   | 0.0516 | 0.0108  | 0.1432         | 0.1879          | 0.0835  | 0.2706  |
| **Pearson**| 0.6885   | 0.1687 | 0.0352  | 0.6156         | 0.7615          | 0.3599  | 0.9168  |
| **CCC**    | 0.6158   | 0.1748 | 0.0365  | 0.5402         | 0.6914          | 0.2563  | 0.9148  |

<img src="Results/Result.png" alt="Result" />

**Ablation Study:**

| # | Experiment | EEG | EOG | Cross-modal | Direction | Gate | MAE | RMSE | Pearson | CCC |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | EEG Only | ✓ | ✗ | ✗ | — | ✗ | 0.1330 | 0.1670 | 0.6733 | 0.6113 |
| 2 | EOG Only | ✗ | ✓ | ✗ | — | ✗ | 0.1852 | 0.2240 | 0.5886 | 0.4495 |
| 3 | Concat | ✓ | ✓ | ✗ | — | ✗ | 0.1364 | 0.1701 | 0.6772 | 0.5989 |
| 4 | Attn Forward | ✓ | ✓ | ✓ | EEG→EOG | ✗ | 0.1369 | 0.1698 | 0.6892 | 0.6197 |
| 5 | Attn Reversed | ✓ | ✓ | ✓ | EOG→EEG | ✗ | 0.1349 | 0.1701 | 0.6921 | 0.6142 |
| 6 | Attn Reversed Gate | ✓ | ✓ | ✓ | EOG→EEG | ✓ | 0.1370 | 0.1712 | 0.6720 | 0.5913 |
| 7 | Full | ✓ | ✓ | ✓ | EEG→EOG | ✓ | 0.1315 | 0.1655 | 0.6885 | 0.6158 |

## Dependencies

| Library | Purpose |
|---|---|
| `numpy` | Numerical computation |
| `scipy` | .mat file loading, kurtosis/correlation statistics |
| `pandas` | CSV result export |
| `mne` | EEG processing (filtering, ICA fitting and reconstruction) |
| `scikit-learn` | Regression metrics (MAE/RMSE) and FastICA |
| `torch` | Model definition and training |
| `safetensors` | Safe model weight serialization |
| `optuna` | Hyperparameter search |

Installation:

```bash
pip install numpy scipy pandas mne scikit-learn torch safetensors optuna
```

## License

### Dataset

The SEED-VIG dataset is copyrighted by the BCMI Laboratory, Shanghai Jiao Tong University, and is limited to **non-commercial research purposes only**. When using this dataset, please cite:

> Wei-Long Zheng and Bao-Liang Lu, *A multimodal approach to estimating vigilance using EEG and forehead EOG*, Journal of Neural Engineering, 14(2): 026017, 2017.

Dataset application page: https://bcmi.sjtu.edu.cn/~seed/seed-vig.html

### Project Code

MIT License — see [LICENSE](LICENSE) file.
