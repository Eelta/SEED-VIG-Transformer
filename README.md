[中文](./README_CN.md) | **English**

# SEED-VIG Transformer

## Description

End-to-end Transformer training on the SEED-VIG dataset for vigilance (PERCLOS) regression prediction under strict LOSO (Leave-One-Subject-Out) cross-validation.

## Usage

1. Download the dataset from: https://huggingface.co/datasets/Curryjiang/SEED-VIG

2. Place raw EEG data in: `Raw_Data\mat_data` and EOG data in: `Raw_Data\perclos_labels`

3. Run `process.py` to batch-process the data and generate `.npy` files

4. Run `Run.py` for hyperparameter optimization, training, validation, and LOSO evaluation

Configuration parameters can be adjusted in `Configs`.

## Model

1. Depthwise separable convolution (EEG/EOG)
2. Dynamic gating + modality random dropout
3. Cross-modal multi-head attention (EEG queries EOG) fusion
4. Self-attention with RoPE encoding
5. FDS (Feature Distribution Smoothing) and LDS (Label Distribution Smoothing) self-calibration
6. Regression head for prediction

## Pipeline

1. Split 23 subjects into 18 for training and 5 for validation; perform hyperparameter search with Optuna (trial=100) — adjust the search range in `Configs\config.json`

2. Iteratively split 23 subjects into 22 for training and 1 for testing (epoch=1) to perform strict LOSO validation, where:

   - **Stage 1:** Full model training with CORAL covariance alignment loss enabled, epoch=200, early stopping
   - **Stage 2:** Freeze the backbone, train only the regression head with FDS and LDS self-calibration enabled, epoch=20, early stopping

## Data

- `eeg.npy` shape: [885000, 4]
- `eog.npy` shape: [885000, 2]
- `label.npy` shape: [885, 1]

Following the original paper, only 4 forehead electrode sites (4, 5, 6, 7) are selected, with ch6 inverted. Quality assessment is performed on raw channel data using kurtosis and correlation metrics. Vertical and horizontal EOG features (VEOf/HEOf) are extracted via channel subtraction and independent component analysis (Fast ICA). An adaptive Pearson correlation coefficient threshold identifies and removes EOG artifacts from EEG independent components to reconstruct clean EEG (manual artifact rejection available via `ica.json`). The cleaned EEG and EOG signals undergo band-pass filtering, outlier clipping, and Z-score normalization, while PERCLOS labels are synchronously extracted, yielding an aligned dataset ready for multimodal regression model training.

## Results

| **LOSO Metric** | **Mean** | **SD** | **SEM** | **95% CI Lower** | **95% CI Upper** | **Min** | **Max** |
| --------------- | -------- | ------ | ------- | ---------------- | ---------------- | ------- | ------- |
| **MAE**         | 0.1287   | 0.0377 | 0.0079  | 0.1124           | 0.1450           | 0.0700  | 0.2130  |
| **RMSE**        | 0.1636   | 0.0442 | 0.0092  | 0.1445           | 0.1827           | 0.0890  | 0.2600  |
| **Pearson**     | 0.6955   | 0.1700 | 0.0354  | 0.6220           | 0.7690           | 0.2340  | 0.9120  |
| **CCC**         | 0.6515   | 0.1739 | 0.0363  | 0.5763           | 0.7267           | 0.2020  | 0.9000  |

<img src="Results/Result.png" alt="Result" />

## Dependencies

| Library         | Purpose                                          |
| --------------- | ------------------------------------------------ |
| `numpy`         | Numerical computation                            |
| `scipy`         | .mat file loading, kurtosis/correlation analysis |
| `pandas`        | CSV result export                                |
| `mne`           | EEG processing (filtering, ICA fitting/reconstruction) |
| `scikit-learn`  | Regression metrics (MAE/RMSE) and FastICA         |
| `torch`         | Model definition and training                    |
| `safetensors`   | Safe model weight serialization                  |
| `optuna`        | Hyperparameter search                            |

Installation:

```bash
pip install numpy scipy pandas mne scikit-learn torch safetensors optuna
```

## License

### Dataset

The SEED-VIG dataset is copyrighted by the BCMI Laboratory, Shanghai Jiao Tong University, and is limited to **non-commercial research purposes only**. When using this dataset, please cite the following paper:

> Wei-Long Zheng and Bao-Liang Lu, *A multimodal approach to estimating vigilance using EEG and forehead EOG*, Journal of Neural Engineering, 14(2): 026017, 2017.

Dataset application page: https://bcmi.sjtu.edu.cn/~seed/seed-vig.html

### Project Code

Apache-2.0 license — see the [LICENSE](LICENSE) file for details.
