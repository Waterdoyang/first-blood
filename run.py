from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from sklearn.metrics import classification_report, confusion_matrix

from src.data import generate_demo_csv, load_and_build_sequences
from src.fd import build_statistics, detect
from src.model import TransformerExtractor
from src.train import evaluate_cls, run_train
from src.utils import make_output_dir, save_json, set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--data", type=str, default="")
    ap.add_argument("--generate-demo", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    set_seed(cfg["seed"])

    csv_path = args.data or cfg["data"]["csv_path"]
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)

    if args.generate_demo and (not os.path.exists(csv_path)):
        generate_demo_csv(csv_path)

    train_seq, val_seq, test_seq = load_and_build_sequences(cfg, csv_path)
    n_features = train_seq[0].shape[-1]
    num_classes = int(max(train_seq[1].max(), val_seq[1].max(), test_seq[1].max()) + 1)

    mcfg = cfg["model"]
    fd_cfg = cfg["fd"]
    model = TransformerExtractor(
        input_dim=n_features,
        d_model=mcfg["d_model"],
        nhead=mcfg["nhead"],
        num_layers=mcfg["num_layers"],
        dim_feedforward=mcfg["dim_feedforward"],
        dropout=mcfg["dropout"],
        feature_dim=fd_cfg["feature_dim"],
        num_classes=num_classes,
    )

    model = run_train(model, train_seq, val_seq, cfg)
    train_eval = evaluate_cls(model, train_seq)
    test_eval = evaluate_cls(model, test_seq)

    stats = build_statistics(train_eval["features"], train_eval["y_true"], alpha=fd_cfg["alpha"])
    t2, spe, alarm = detect(stats, test_eval["features"])
    y_bin = (test_eval["y_true"] != 0).astype(int)

    det_acc = float((alarm == y_bin).mean())
    out = make_output_dir()
    torch.save(model.state_dict(), os.path.join(out, "model.pt"))

    metrics = {
        "classification": {"acc": test_eval["acc"], "f1_macro": test_eval["f1_macro"]},
        "detection": {"binary_accuracy": det_acc},
    }
    save_json(os.path.join(out, "metrics.json"), metrics)
    save_json(os.path.join(out, "thresholds.json"), {"t2_thr": stats["t2_thr"], "spe_thr": stats["spe_thr"]})

    plt.figure(figsize=(10, 4))
    plt.plot(t2, label="T2")
    plt.axhline(stats["t2_thr"], color="r", linestyle="--", label="T2 threshold")
    plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(out, "plots", "t2.png")); plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(spe, label="SPE")
    plt.axhline(stats["spe_thr"], color="r", linestyle="--", label="SPE threshold")
    plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(out, "plots", "spe.png")); plt.close()

    print("Done. Outputs:", out)


if __name__ == "__main__":
    main()
