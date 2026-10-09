"""
Data loading and preprocessing for UCR/UEA time series datasets.

Datasets (Table 1):
  Univariate:  CBF, TwoLeadECG, GunPoint, Earthquakes, Coffee, ItalyPowerDemand
  Multivariate: Cricket, Epilepsy

Uses `aeon` for automatic downloading from the UCR/UEA archives.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
import sys

# ── Auto-download via aeon ───────────────────────────────────────────────────
try:
    from aeon.datasets import load_classification
    HAS_AEON = True
except ImportError:
    HAS_AEON = False
    print("[WARNING] aeon not installed. Run: pip install aeon")


class TimeSeriesDataset(Dataset):
    """
    PyTorch Dataset wrapper for time series classification data.
    
    Stores data as tensors of shape (n_samples, n_channels, seq_len)
    and integer class labels.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        Args:
            X: numpy array of shape (n_samples, n_channels, seq_len)
            y: numpy array of integer labels (n_samples,)
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def _cache_path(name: str) -> str:
    try:
        from soft_dtw_cfe.config import DATA_DIR
    except ImportError:
        from config import DATA_DIR
    return os.path.join(DATA_DIR, f"{name}.npz")


def _normalize_channels(X_train: np.ndarray, X_test: np.ndarray):
    """Z-score normalize each channel independently using training statistics."""
    for ch in range(X_train.shape[1]):
        mean = float(X_train[:, ch, :].mean())
        std = float(X_train[:, ch, :].std())
        if std < 1e-8:
            std = 1.0
        X_train[:, ch, :] = (X_train[:, ch, :] - mean) / std
        X_test[:, ch, :] = (X_test[:, ch, :] - mean) / std
    return X_train, X_test


def load_physionet_mitbih(seq_len: int = 5400, n_per_record: int = 4):
    """
    Load long multivariate ECG time series from PhysioNet MIT-BIH Arrhythmia Database.
    
    Channels: d=2 (MLII, V1/V5)
    Length: T=5400 (15 seconds at 360 Hz)
    Classes: 
      0: Normal Sinus Rhythm (records 100, 101, 103, 105, 112, 113, 115, 117, 121, 122)
      1: Ventricular Arrhythmia (records 106, 119, 200, 201, 203, 208, 210, 213, 215, 221)
    """
    cache = _cache_path("PhysioNet_MITBIH")
    if os.path.exists(cache):
        print(f"[DATA] Loading cached PhysioNet_MITBIH from {cache}")
        data = np.load(cache)
        X_train, y_train = data["X_train"], data["y_train"]
        X_test, y_test = data["X_test"], data["y_test"]
    else:
        print("[DATA] Downloading / extracting PhysioNet MIT-BIH records...")
        import wfdb
        normal_recs = ["100", "101", "103", "105", "112", "113", "115", "117", "121", "122"]
        arrhyth_recs = ["106", "119", "200", "201", "203", "208", "210", "213", "215", "221"]

        X_list, y_list = [], []

        for rec_id in normal_recs:
            try:
                rec = wfdb.rdrecord(rec_id, pn_dir="mitdb", sampto=seq_len * n_per_record)
                sig = rec.p_signal.astype(np.float32)  # (N, 2)
                for i in range(n_per_record):
                    seg = sig[i * seq_len:(i + 1) * seq_len]
                    if len(seg) == seq_len:
                        X_list.append(seg.T)  # (2, T)
                        y_list.append(0)
            except Exception as e:
                print(f"  [WARN] Record {rec_id} error: {e}")

        for rec_id in arrhyth_recs:
            try:
                rec = wfdb.rdrecord(rec_id, pn_dir="mitdb", sampto=seq_len * n_per_record)
                sig = rec.p_signal.astype(np.float32)
                for i in range(n_per_record):
                    seg = sig[i * seq_len:(i + 1) * seq_len]
                    if len(seg) == seq_len:
                        X_list.append(seg.T)
                        y_list.append(1)
            except Exception as e:
                print(f"  [WARN] Record {rec_id} error: {e}")

        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.int64)

        # Stratified train/test split (70/30)
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        X_train, X_test = _normalize_channels(X_train, X_test)
        np.savez_compressed(cache, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

    metadata = {
        "n_classes": 2,
        "seq_len": X_train.shape[2],
        "n_channels": X_train.shape[1],
        "n_train": len(X_train),
        "n_test": len(X_test),
        "label_map": {0: "Normal", 1: "Arrhythmia"},
    }
    print(f"  -> n_train={metadata['n_train']}, n_test={metadata['n_test']}, "
          f"T={metadata['seq_len']}, d={metadata['n_channels']}, c={metadata['n_classes']}")
    return X_train, y_train, X_test, y_test, metadata


def load_uci_eeg_eyestate(seq_len: int = 5000, stride: int = 500):
    """
    Load long multivariate EEG time series from UCI EEG Eye State.
    
    Channels: d=14 (AF3, F7, F3, FC5, T7, P7, O1, O2, P8, T8, FC6, F4, F8, AF4)
    Length: T=5000 time steps
    Classes: 0 (Eye Open), 1 (Eye Closed)
    """
    cache = _cache_path("UCI_EEGEyeState")
    if os.path.exists(cache):
        print(f"[DATA] Loading cached UCI_EEGEyeState from {cache}")
        data = np.load(cache)
        X_train, y_train = data["X_train"], data["y_train"]
        X_test, y_test = data["X_test"], data["y_test"]
    else:
        print("[DATA] Downloading UCI EEG Eye State...")
        import urllib.request
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00264/EEG%20Eye%20State.arff"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as r:
            lines = r.read().decode("utf-8", errors="ignore").splitlines()

        data_lines = [l for l in lines if l and not l.startswith("@")]
        raw = np.array([[float(x) for x in l.split(",")] for l in data_lines], dtype=np.float32)

        signals = raw[:, :-1]  # (14980, 14)
        labels = raw[:, -1].astype(np.int64)

        X_windows, y_windows = [], []
        for start in range(0, len(signals) - seq_len + 1, stride):
            window = signals[start:start + seq_len].T  # (14, T)
            lbl = int(np.round(labels[start:start + seq_len].mean()))
            X_windows.append(window)
            y_windows.append(lbl)

        X = np.array(X_windows, dtype=np.float32)
        y = np.array(y_windows, dtype=np.int64)

        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        X_train, X_test = _normalize_channels(X_train, X_test)
        np.savez_compressed(cache, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

    metadata = {
        "n_classes": 2,
        "seq_len": X_train.shape[2],
        "n_channels": X_train.shape[1],
        "n_train": len(X_train),
        "n_test": len(X_test),
        "label_map": {0: "EyeOpen", 1: "EyeClosed"},
    }
    print(f"  -> n_train={metadata['n_train']}, n_test={metadata['n_test']}, "
          f"T={metadata['seq_len']}, d={metadata['n_channels']}, c={metadata['n_classes']}")
    return X_train, y_train, X_test, y_test, metadata


def load_coupled_lorenz_dynamics(seq_len: int = 5000, n_samples_per_class: int = 25):
    """
    Generate synthetic long multivariate nonlinear dynamical system (Coupled Lorenz Attractors).
    
    Channels: d=6 (x1, y1, z1, x2, y2, z2)
    Length: T=5000
    Classes:
      0: Synchronous Chaos (coupling kappa=2.5, rho=28.0)
      1: Desynchronous Chaos (coupling kappa=0.15, rho=28.0)
      2: Periodic Limit Cycle (rho=14.0, below Hopf bifurcation)
    """
    cache = _cache_path("CoupledLorenz_Dynamics")
    if os.path.exists(cache):
        print(f"[DATA] Loading cached CoupledLorenz_Dynamics from {cache}")
        data = np.load(cache)
        X_train, y_train = data["X_train"], data["y_train"]
        X_test, y_test = data["X_test"], data["y_test"]
    else:
        print("[DATA] Generating Coupled Lorenz continuous dynamics (T=5000, d=6)...")
        dt = 0.01
        sigma = 10.0
        beta = 8.0 / 3.0

        configs = [
            (2.5, 28.0, 0),   # Synchronous chaos
            (0.15, 28.0, 1),  # Desynchronous chaos
            (0.5, 14.0, 2),   # Periodic / Limit cycle
        ]

        np.random.seed(42)
        X_list, y_list = [], []

        for kappa, rho, label in configs:
            for s in range(n_samples_per_class):
                # Random initial condition
                state = np.random.randn(6) * 2.0 + np.array([1.0, 1.0, 20.0, 1.2, 0.8, 20.2])
                traj = np.zeros((seq_len, 6), dtype=np.float32)

                # Warmup 500 steps to reach attractor
                for _ in range(500):
                    x1, y1, z1, x2, y2, z2 = state
                    dx1 = sigma * (y1 - x1) + kappa * (x2 - x1)
                    dy1 = x1 * (rho - z1) - y1
                    dz1 = x1 * y1 - beta * z1
                    dx2 = sigma * (y2 - x2) + kappa * (x1 - x2)
                    dy2 = x2 * (rho - z2) - y2
                    dz2 = x2 * y2 - beta * z2
                    state = state + dt * np.array([dx1, dy1, dz1, dx2, dy2, dz2])

                for t in range(seq_len):
                    x1, y1, z1, x2, y2, z2 = state
                    dx1 = sigma * (y1 - x1) + kappa * (x2 - x1)
                    dy1 = x1 * (rho - z1) - y1
                    dz1 = x1 * y1 - beta * z1
                    dx2 = sigma * (y2 - x2) + kappa * (x1 - x2)
                    dy2 = x2 * (rho - z2) - y2
                    dz2 = x2 * y2 - beta * z2
                    state = state + dt * np.array([dx1, dy1, dz1, dx2, dy2, dz2])
                    traj[t] = state

                X_list.append(traj.T)  # (6, T)
                y_list.append(label)

        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.int64)

        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        X_train, X_test = _normalize_channels(X_train, X_test)
        np.savez_compressed(cache, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

    metadata = {
        "n_classes": 3,
        "seq_len": X_train.shape[2],
        "n_channels": X_train.shape[1],
        "n_train": len(X_train),
        "n_test": len(X_test),
        "label_map": {0: "SyncChaos", 1: "DesyncChaos", 2: "LimitCycle"},
    }
    print(f"  -> n_train={metadata['n_train']}, n_test={metadata['n_test']}, "
          f"T={metadata['seq_len']}, d={metadata['n_channels']}, c={metadata['n_classes']}")
    return X_train, y_train, X_test, y_test, metadata


def load_ptbxl(
    task: str = "binary_norm_mi",
    n_samples_per_class: int = 50,
    downsample: int = 4,
    test_size: float = 0.3,
    random_state: int = 42,
):
    """
    Load real 12-lead clinical ECG time series from PhysioNet PTB-XL database.
    
    Channels: d=12 (I, II, III, AVR, AVL, AVF, V1, V2, V3, V4, V5, V6)
    Length: T = 1000 // downsample (default T=250 for fast Soft-DTW, downsample=1 for full T=1000)
    Classes:
      binary_norm_mi:
        0: Normal (NORM)
        1: Myocardial Infarction (MI)
      multiclass_5:
        0: NORM, 1: MI, 2: STTC, 3: CD, 4: HYP
    """
    eff_T = 1000 // max(downsample, 1)
    cache = _cache_path(f"PTB_XL_{task}_n{n_samples_per_class}_T{eff_T}")
    label_map = {0: "Normal", 1: "Myocardial_Infarction"} if task == "binary_norm_mi" else {
        0: "NORM", 1: "MI", 2: "STTC", 3: "CD", 4: "HYP"
    }

    if os.path.exists(cache):
        print(f"[DATA] Loading cached PTB-XL from {cache}")
        data = np.load(cache)
        X_train, y_train = data["X_train"], data["y_train"]
        X_test, y_test = data["X_test"], data["y_test"]
    else:
        print(f"[DATA] Preparing PTB-XL 12-lead ECG dataset ({task}, {n_samples_per_class} samples/class, T={eff_T})...")
        import pandas as pd
        import ast
        from concurrent.futures import ThreadPoolExecutor

        try:
            from soft_dtw_cfe.config import DATA_DIR
        except ImportError:
            from config import DATA_DIR

        csv_path = os.path.join(DATA_DIR, "ptbxl_database.csv")
        scp_path = os.path.join(DATA_DIR, "scp_statements.csv")

        # Download metadata CSVs if not already present
        if not os.path.exists(csv_path):
            import urllib.request
            print("  Downloading PTB-XL database metadata...")
            req = urllib.request.Request("https://physionet.org/files/ptb-xl/1.0.3/ptbxl_database.csv", headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as r, open(csv_path, "wb") as f:
                f.write(r.read())

        if not os.path.exists(scp_path):
            import urllib.request
            req = urllib.request.Request("https://physionet.org/files/ptb-xl/1.0.3/scp_statements.csv", headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as r, open(scp_path, "wb") as f:
                f.write(r.read())

        df = pd.read_csv(csv_path)
        df_scp = pd.read_csv(scp_path, index_col=0)
        diag_map = df_scp[df_scp["diagnostic"] == 1.0]["diagnostic_class"].to_dict()

        def get_classes(s):
            d = ast.literal_eval(s) if isinstance(s, str) else {}
            return set(diag_map[k] for k in d if k in diag_map)

        df["classes"] = df["scp_codes"].apply(get_classes)

        if task == "binary_norm_mi":
            norm_files = df[df["classes"] == {"NORM"}]["filename_lr"].tolist()[:n_samples_per_class]
            mi_files = df[df["classes"] == {"MI"}]["filename_lr"].tolist()[:n_samples_per_class]
            file_label_pairs = [(f, 0) for f in norm_files] + [(f, 1) for f in mi_files]
        else:
            classes_order = ["NORM", "MI", "STTC", "CD", "HYP"]
            file_label_pairs = []
            for idx, c in enumerate(classes_order):
                c_files = df[df["classes"] == {c}]["filename_lr"].tolist()[:n_samples_per_class]
                file_label_pairs.extend([(f, idx) for f in c_files])

        import wfdb
        def _fetch_record(item):
            fpath, lbl = item
            folder, fname = os.path.split(fpath)
            rec = wfdb.rdrecord(fname, pn_dir=f"ptb-xl/1.0.3/{folder}/")
            sig = rec.p_signal.T  # (12, 1000)
            if downsample > 1:
                sig = sig[:, ::downsample]
            return sig.astype(np.float32), lbl

        print(f"  Downloading and processing {len(file_label_pairs)} 12-lead ECG records via ThreadPool...")
        with ThreadPoolExecutor(max_workers=10) as executor:
            fetched = list(executor.map(_fetch_record, file_label_pairs))

        X = np.array([f[0] for f in fetched], dtype=np.float32)
        y = np.array([f[1] for f in fetched], dtype=np.int64)

        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        X_train, X_test = _normalize_channels(X_train, X_test)
        np.savez_compressed(cache, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

    metadata = {
        "n_classes": len(np.unique(y_train)),
        "seq_len": X_train.shape[2],
        "n_channels": X_train.shape[1],
        "n_train": len(X_train),
        "n_test": len(X_test),
        "label_map": label_map,
        "leads": ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"],
    }
    print(f"  -> n_train={metadata['n_train']}, n_test={metadata['n_test']}, "
          f"T={metadata['seq_len']}, d={metadata['n_channels']}, c={metadata['n_classes']}")
    return X_train, y_train, X_test, y_test, metadata


def load_dataset(name: str):
    """
    Load a dataset by name from UCR/UEA or extended long multivariate archives.
    
    Supports:
      - PTB_XL: Real 12-lead clinical ECG (d=12, T=250/1000)
      - PhysioNet_MITBIH: Real 2-lead ECG (d=2)
      - UCI_EEGEyeState: Real 14-channel EEG (d=14, T=5000)
      - CoupledLorenz_Dynamics: 6-channel dynamical system (d=6, T=5000)
      - UCR/UEA datasets via aeon (Epilepsy, BasicMotions, etc.)
    """
    print(f"[DATA] Loading dataset: {name} ...")

    if name == "PTB_XL" or name.startswith("PTB_XL"):
        return load_ptbxl()
    elif name == "PhysioNet_MITBIH":
        return load_physionet_mitbih()
    elif name == "UCI_EEGEyeState":
        return load_uci_eeg_eyestate()
    elif name == "CoupledLorenz_Dynamics":
        return load_coupled_lorenz_dynamics()

    if not HAS_AEON:
        raise ImportError("aeon is required for UCR/UEA dataset loading. Install via: pip install aeon")

    # Load from aeon — returns (X, y) where X has shape (n, n_channels, seq_len)
    X_train, y_train = load_classification(name, split="train")
    X_test, y_test = load_classification(name, split="test")

    # aeon returns X as np.ndarray of shape (n_samples, n_channels, seq_len)
    all_labels = sorted(set(y_train.tolist()) | set(y_test.tolist()))
    label_map = {label: idx for idx, label in enumerate(all_labels)}
    y_train = np.array([label_map[l] for l in y_train])
    y_test = np.array([label_map[l] for l in y_test])

    X_train = np.nan_to_num(X_train, nan=0.0).astype(np.float32)
    X_test = np.nan_to_num(X_test, nan=0.0).astype(np.float32)
    X_train, X_test = _normalize_channels(X_train, X_test)

    n_classes = len(all_labels)
    seq_len = X_train.shape[2]
    n_channels = X_train.shape[1]

    metadata = {
        "n_classes": n_classes,
        "seq_len": seq_len,
        "n_channels": n_channels,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "label_map": label_map,
    }

    print(f"  -> n_train={metadata['n_train']}, n_test={metadata['n_test']}, "
          f"T={seq_len}, d={n_channels}, c={n_classes}")

    return X_train, y_train, X_test, y_test, metadata


def get_dataloaders(X_train, y_train, X_test, y_test, batch_size=32):
    """Create PyTorch DataLoaders from numpy arrays."""
    train_ds = TimeSeriesDataset(X_train, y_train)
    test_ds = TimeSeriesDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    return train_loader, test_loader


def get_target_class_samples(X_train, y_train, target_class, device="cpu"):
    """
    Get all training samples belonging to a specific class.
    
    Returns:
        torch.Tensor of shape (n_target, n_channels, seq_len)
    """
    mask = y_train == target_class
    X_target = X_train[mask]
    return torch.tensor(X_target, dtype=torch.float32).to(device)

