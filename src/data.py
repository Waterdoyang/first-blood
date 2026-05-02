from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class SequenceDataset:
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = x.astype(np.float32)
        self.y = y.astype(np.int64)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def generate_demo_csv(path: str, n: int = 8000) -> None:
    t = pd.date_range("2025-01-01", periods=n, freq="5min")
    base = np.linspace(0, 30, n)
    rng = np.random.default_rng(42)

    df = pd.DataFrame({
        "timestamp": t,
        "supply_air_temp": 14 + 2 * np.sin(base) + rng.normal(0, 0.25, n),
        "return_air_temp": 24 + 1.8 * np.sin(base / 1.3) + rng.normal(0, 0.3, n),
        "outdoor_air_temp": 20 + 8 * np.sin(base / 6) + rng.normal(0, 0.6, n),
        "supply_air_humidity": 45 + 7 * np.sin(base / 2) + rng.normal(0, 1.2, n),
        "fan_speed": 55 + 20 * np.abs(np.sin(base / 2.5)) + rng.normal(0, 1.0, n),
        "damper_position": 40 + 30 * np.sin(base / 4) + rng.normal(0, 1.5, n),
        "cooling_valve": 35 + 30 * np.sin(base / 3.5) + rng.normal(0, 2.0, n),
        "heating_valve": 15 + 10 * np.cos(base / 3.0) + rng.normal(0, 1.5, n),
        "static_pressure": 300 + 15 * np.sin(base / 2.2) + rng.normal(0, 2.0, n),
    })

    label = np.zeros(n, dtype=int)
    # 注入故障段：1=风阀卡滞, 2=冷却阀泄漏, 3=风机效率下降
    for st, ed, k in [(2500, 3300, 1), (4300, 5100, 2), (6200, 7000, 3)]:
        label[st:ed] = k

    df.loc[label == 1, "damper_position"] += 20
    df.loc[label == 2, "cooling_valve"] += 18
    df.loc[label == 3, "fan_speed"] -= 15
    df["fault_label"] = label
    df.to_csv(path, index=False)


def load_and_build_sequences(cfg: dict, csv_path: str):
    dcfg = cfg["data"]
    df = pd.read_csv(csv_path)
    df[dcfg["time_col"]] = pd.to_datetime(df[dcfg["time_col"]])
    df = df.sort_values(dcfg["time_col"]).reset_index(drop=True)

    x = df[dcfg["feature_cols"]].values
    y = df[dcfg["label_col"]].values

    n = len(df)
    n_train = int(n * dcfg["train_ratio"])
    n_val = int(n * dcfg["val_ratio"])

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x[:n_train])
    x_val = scaler.transform(x[n_train:n_train + n_val])
    x_test = scaler.transform(x[n_train + n_val:])

    y_train = y[:n_train]
    y_val = y[n_train:n_train + n_val]
    y_test = y[n_train + n_val:]

    def make_seq(xs, ys, seq_len, stride):
        xseq, yseq = [], []
        for i in range(0, len(xs) - seq_len + 1, stride):
            j = i + seq_len
            xseq.append(xs[i:j])
            yseq.append(int(np.bincount(ys[i:j]).argmax()))
        return np.array(xseq), np.array(yseq)

    seq_len = dcfg["sequence_length"]
    stride = dcfg["stride"]
    train_seq = make_seq(x_train, y_train, seq_len, stride)
    val_seq = make_seq(x_val, y_val, seq_len, stride)
    test_seq = make_seq(x_test, y_test, seq_len, stride)

    return train_seq, val_seq, test_seq
