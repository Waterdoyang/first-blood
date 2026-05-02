from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

from .data import SequenceDataset


def run_train(model, train_seq, val_seq, cfg):
    tcfg = cfg["train"]
    device = torch.device("cpu")
    model.to(device)

    train_ds = SequenceDataset(*train_seq)
    val_ds = SequenceDataset(*val_seq)
    train_loader = DataLoader(train_ds, batch_size=tcfg["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=tcfg["batch_size"], shuffle=False)

    criterion = torch.nn.CrossEntropyLoss()
    optim = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])

    best = -1
    best_state = None
    for _ in range(tcfg["epochs"]):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            _, logits = model(xb)
            loss = criterion(logits, yb)
            optim.zero_grad()
            loss.backward()
            optim.step()

        model.eval()
        pred, true = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                _, logits = model(xb.to(device))
                pred.extend(logits.argmax(1).cpu().numpy().tolist())
                true.extend(yb.numpy().tolist())
        f1 = f1_score(true, pred, average="macro", zero_division=0)
        if f1 > best:
            best = f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return model


def evaluate_cls(model, seq):
    ds = SequenceDataset(*seq)
    loader = DataLoader(ds, batch_size=256, shuffle=False)
    model.eval()
    pred, true = [], []
    feats = []
    with torch.no_grad():
        for xb, yb in loader:
            f, logits = model(xb)
            feats.append(f.numpy())
            pred.extend(logits.argmax(1).numpy().tolist())
            true.extend(yb.numpy().tolist())
    return {
        "acc": float(accuracy_score(true, pred)),
        "f1_macro": float(f1_score(true, pred, average="macro", zero_division=0)),
        "y_true": np.array(true),
        "y_pred": np.array(pred),
        "features": np.vstack(feats),
    }
