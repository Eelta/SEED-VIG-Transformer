from pathlib import Path
import json
import numpy as np
from scipy.io import loadmat
from scipy.stats import kurtosis, pearsonr
import mne
import warnings
import sys

warnings.filterwarnings('ignore', category=RuntimeWarning)

input_dir = Path('mat_data')
output_dir = Path('../Data')
output_dir.mkdir(parents=True, exist_ok=True)

# 加载ica去除项
_ica_config = None
_ica_json_path = Path(__file__).resolve().parent.parent / 'Configs' / 'ica.json'
if _ica_json_path.exists():
    with open(_ica_json_path, 'r', encoding='utf-8') as f:
        _ica_config = json.load(f)

# 通道索引
CH_IDX = [3, 4, 5, 6]
CH_NAMES = ['Ch4', 'Ch5', 'Ch6', 'Ch7']


def _get_subject_id(mat_path):
    stem = mat_path.stem
    digits = ''.join(ch for ch in stem if ch.isdigit())
    if digits:
        return digits
    return stem


def rename_file(root):
    files = sorted(list(root.glob('*.mat')), key=lambda x: int(_get_subject_id(x)))
    temp_files = []
    for i, f in enumerate(files, start=1):
        temp_path = root / f"temp_{i}_{f.name}"
        f.rename(temp_path)
        temp_files.append(temp_path)
    for i, tf in enumerate(temp_files, start=1):
        tf.rename(root / f"{i}.mat")


def _fit_ica_transform(ica, X, tag='ICA'):
    """sklearn FastICA + 收敛检测"""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        S = ica.fit_transform(X)
        converged = True
        for warning in w:
            if 'convergence' in str(warning.message).lower():
                converged = False
                break
    if not converged:
        print(f"    [Warning] {tag} not converged, retrying...")
        ica.max_iter = 5000
        ica.tol = 1e-3
        S = ica.fit_transform(X)
    return S, ica


def _extract_eog(ch4, ch5, ch6, ch7):
    from sklearn.decomposition import FastICA
    heof = ch5 - ch6
    X_veo = np.vstack([ch4, ch7])
    S_veo, _ = _fit_ica_transform(
        FastICA(n_components=2, random_state=42, max_iter=2000),
        X_veo.T, tag='VEOf')
    S_veo = S_veo.T
    k0, k1 = kurtosis(S_veo[0]), kurtosis(S_veo[1])
    veof = S_veo[0] if k0 > k1 else S_veo[1]
    return np.vstack([veof, heof])


def _quality_check(ch_data):
    diag = {}
    for name, sig in zip(CH_NAMES, [ch_data[i] for i in CH_IDX]):
        diag[name] = {
            'kurt': float(kurtosis(sig)),
            'max_abs': float(np.max(np.abs(sig))),
            'std': float(np.std(sig)),
        }
    ch_matrix = np.vstack([ch_data[i] for i in CH_IDX])
    corr = np.corrcoef(ch_matrix)
    diag['avg_corr'] = float((corr.sum() - 4) / 12)
    flags = []
    avg_kurt = np.mean([diag[n]['kurt'] for n in CH_NAMES])
    if avg_kurt < 10:
        flags.append('flat_signal')
    for name in CH_NAMES:
        if diag[name]['kurt'] > 300:
            flags.append(f'{name}_extreme_kurt({diag[name]["kurt"]:.0f})')
    if diag['avg_corr'] > 0.90:
        flags.append(f'high_ch_corr({diag["avg_corr"]:.2f})')
    diag['flags'] = flags
    return diag


def _auto_threshold(scores):
    scores = np.asarray(scores)
    if np.max(scores) < 0.15:
        return [], 0.15

    sorted_idx = np.argsort(scores)[::-1]
    sorted_val = scores[sorted_idx]
    gaps = [sorted_val[i] - sorted_val[i + 1] for i in range(len(sorted_val) - 1)]

    if not gaps:
        return [], 0.15

    max_gap_idx = int(np.argmax(gaps))
    threshold = (sorted_val[max_gap_idx] + sorted_val[max_gap_idx + 1]) / 2
    threshold = max(threshold, 0.10)

    eog_ic_idx = [int(i) for i in range(len(scores)) if scores[i] > threshold]

    if len(eog_ic_idx) > 2:
        idx_by_score = sorted(eog_ic_idx, key=lambda i: scores[i], reverse=True)
        eog_ic_idx = idx_by_score[:2]
        kept_min = min(scores[i] for i in eog_ic_idx)
        threshold = max(threshold, kept_min - 0.01)

    return eog_ic_idx, threshold


def process_subject(mat_path):
    subject_id = _get_subject_id(mat_path)
    print(f"\n{'=' * 55}")
    print(f"  Subject {subject_id}")
    print(f"{'=' * 55}")

    mat_data = loadmat(mat_path)
    forehead_raw = mat_data['EOG']['eog'][0, 0].astype(np.float64)

    if forehead_raw.shape[0] > forehead_raw.shape[1]:
        forehead_raw = forehead_raw.T
    if forehead_raw.shape[0] < 7:
        print(f"  [SKIP] only {forehead_raw.shape[0]} channels (< 7)")
        return None

    ch_data = {i: forehead_raw[i, :] for i in range(forehead_raw.shape[0])}
    qc = _quality_check(ch_data)
    print(f"  Quality: avg_kurt={np.mean([qc[n]['kurt'] for n in CH_NAMES]):.0f}, "
          f"avg_ch_corr={qc['avg_corr']:.3f}")
    if qc['flags']:
        print(f"  ⚠️  {', '.join(qc['flags'])}")

    ch4 = forehead_raw[3, :]
    ch5 = forehead_raw[4, :]
    ch6 = forehead_raw[5, :]
    ch7 = forehead_raw[6, :]

    # EOG 提取
    eog_extracted = _extract_eog(ch4, ch5, ch6, ch7)
    veof, heof = eog_extracted[0, :], eog_extracted[1, :]

    # MNE FastICA
    X_eeg = np.vstack([ch4, ch5, -ch6, ch7])
    ch_names_eeg = ['FP_Ch4', 'FP_Ch5', 'FP_Ch6_inv', 'FP_Ch7']

    info = mne.create_info(ch_names=ch_names_eeg, sfreq=125, ch_types='eeg')
    raw = mne.io.RawArray(X_eeg, info, verbose=False)
    # 1.0Hz 契合论文 2.3.4 节实验参数
    raw.filter(l_freq=1.0, h_freq=None, fir_design='firwin',
               skip_by_annotation='edge', verbose=False)

    ica = mne.preprocessing.ICA(
        n_components=4, method='fastica',
        random_state=42, max_iter=2000, verbose=False)
    ica.fit(raw)

    ic_sources = ica.get_sources(raw).get_data()
    eog_ref = np.vstack([veof, heof]).T[:ic_sources.shape[1], :]

    eog_scores = []
    for i in range(4):
        best = 0.0
        for j in range(2):
            if np.std(ic_sources[i]) > 1e-8 and np.std(eog_ref[:, j]) > 1e-8:
                r, _ = pearsonr(ic_sources[i], eog_ref[:, j])
                best = max(best, abs(r))
        eog_scores.append(best)

    ic_kurts = [float(kurtosis(ic_sources[i])) for i in range(4)]

    # 选择 EOG 伪影成分
    print(f"\n  {'─' * 55}")
    print(f"  Subject {subject_id} — ICA Component Review")
    print(f"  {'─' * 55}")
    print(f"  {'IC':<6} {'EOG corr':>10} {'Kurtosis':>10}  Likely type")
    print(f"  {'─' * 55}")
    for i in range(4):
        s, k = eog_scores[i], ic_kurts[i]
        if s > 0.3:
            t = 'EOG (blink/saccade)'
        elif k > 10:
            t = 'artifact (high kurt)'
        else:
            t = 'EEG'
        print(f"  IC{i:<5} {s:>10.3f} {k:>10.1f}  {t}")
    print(f"  {'─' * 55}")

    eog_ic_idx, auto_thr = _auto_threshold(eog_scores)

    # 如果 ica.json 存在且 ica_auto=0，则使用文件中指定的 exclude 列表
    ica_auto = True
    if _ica_config is not None:
        ica_auto = _ica_config.get('ica_auto', 1) == 1
        if not ica_auto:
            sub_data = _ica_config.get('subjects', {}).get(str(subject_id))
            if sub_data is not None and 'exclude' in sub_data:
                ica_exclude = sub_data['exclude']
                print(f"  ica_auto=0 → use ica.json exclude: {ica_exclude}")
                eog_ic_idx = ica_exclude

    if ica_auto:
        print(f"  Adaptive threshold: {auto_thr:.3f} "
              f"(largest gap: {eog_ic_idx if eog_ic_idx else 'none removed'})")

    print(f"  Subject {subject_id} — use component {eog_ic_idx} reconstruct signal...")

    # 重构 EEG（移除 EOG 成分）
    raw_clean = ica.apply(raw.copy(), exclude=eog_ic_idx, verbose=False)
    clean_data = raw_clean.get_data()
    clean_data[2, :] = -clean_data[2, :]

    ica_diag = {
        'eog_scores': np.array(eog_scores),
        'max_score': max(eog_scores) if eog_scores else 0,
        'n_removed': len(eog_ic_idx),
        'eog_ic_idx': eog_ic_idx,
        'threshold': auto_thr,
    }
    return subject_id, clean_data, eog_extracted, ica_diag


def save_processed(subject_id, eeg_clean, eog_extracted):
    sfreq = 125

    eeg_info = mne.create_info(
        ch_names=['FP_Ch4', 'FP_Ch5', 'FP_Ch6', 'FP_Ch7'],
        sfreq=sfreq, ch_types='eeg')
    eeg_mne = mne.io.RawArray(eeg_clean, eeg_info, verbose=False)

    # 1.0Hz - 45HZ滤波，与前置 ICA 处理保持频带一致
    eeg_mne.filter(l_freq=1.0, h_freq=45.0, fir_design='firwin',
                   skip_by_annotation='edge', verbose=False, picks='all')
    eeg_processed = eeg_mne.get_data().T.astype(np.float32)

    # 异常值裁剪（Clip）：先进行物理幅值上的异常切除，再做 Z-score
    eeg_std = np.std(eeg_processed, axis=0) + 1e-8
    eeg_mean = np.mean(eeg_processed, axis=0)
    # 裁剪掉偏离均值 5 倍标准差以上的突发极端噪声点
    eeg_processed = np.clip(eeg_processed, eeg_mean - 5 * eeg_std, eeg_mean + 5 * eeg_std)

    # Z-score 归一化
    eeg_processed = ((eeg_processed - np.mean(eeg_processed, axis=0)) /
                     (np.std(eeg_processed, axis=0) + 1e-8))

    np.save(output_dir / f"{subject_id}_eeg.npy", eeg_processed)

    eog_info = mne.create_info(
        ch_names=['VEOf', 'HEOf'], sfreq=sfreq, ch_types='eog')
    eog_mne = mne.io.RawArray(eog_extracted, eog_info, verbose=False)
    eog_mne.filter(l_freq=1.0, h_freq=40.0, fir_design='firwin',
                   skip_by_annotation='edge', verbose=False, picks='all')
    eog_processed = eog_mne.get_data().T.astype(np.float32)

    eog_std = np.std(eog_processed, axis=0) + 1e-8
    eog_mean = np.mean(eog_processed, axis=0)
    eog_processed = np.clip(eog_processed, eog_mean - 5 * eog_std, eog_mean + 5 * eog_std)

    eog_processed = ((eog_processed - np.mean(eog_processed, axis=0)) /
                     (np.std(eog_processed, axis=0) + 1e-8))

    np.save(output_dir / f"{subject_id}_eog.npy", eog_processed)

    print(f"  ✅ Saved: {subject_id}_eeg.npy, {subject_id}_eog.npy")


def get_label():
    label_dir = Path('perclos_labels')
    for mat_path in sorted(label_dir.glob('*.mat')):
        subject_id = _get_subject_id(mat_path)
        label = loadmat(mat_path)['perclos'].astype(np.float32)
        np.save(output_dir / f"{subject_id}_label.npy", label)


def data_check():
    data_dir = Path('../Data')
    subject_ids = sorted(set(
        f.name.split('_')[0] for f in data_dir.glob('*_eeg.npy')),
        key=lambda x: int(''.join(ch for ch in x if ch.isdigit()) or '0'))
    print(f"\n  Found {len(subject_ids)} subjects\n")
    for sub_id in subject_ids:
        print(f"  {'─' * 45}")
        print(f"  Subject {sub_id}")
        for suffix in ['eeg', 'eog', 'label']:
            fp = data_dir / f"{sub_id}_{suffix}.npy"
            if fp.exists():
                d = np.load(fp)
                print(f"  [{suffix.upper()}] shape={d.shape}, mean={d.mean():.4f}, "
                      f"std={d.std():.4f}, |max|={np.max(np.abs(d)):.2f}, NaN={np.isnan(d).any()}")
            else:
                print(f"  [{suffix.upper()}] ❌ MISSING")


if __name__ == '__main__':
    mat_files = sorted(input_dir.glob('*.mat'))
    if not mat_files:
        print(f"No .mat files found in {input_dir}/")
        sys.exit(1)

    print(f"Found {len(mat_files)} .mat files\n")

    for mat_path in mat_files:
        try:
            result = process_subject(mat_path)
        except Exception as e:
            print(f"  ❌ Error: {e}")
            import traceback

            traceback.print_exc()
            continue

        if result is None:
            continue

        subject_id, eeg_clean, eog_extracted, ica_diag = result

        scores = ica_diag['eog_scores']
        print(f"  ICA scores: " +
              ', '.join(f'IC{i}={s:.3f}' for i, s in enumerate(scores)))
        print(f"  Removed {ica_diag['n_removed']} IC(s): {ica_diag['eog_ic_idx']}")

        save_processed(subject_id, eeg_clean, eog_extracted)

    get_label()
    data_check()
