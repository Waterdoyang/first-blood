from __future__ import annotations

import numpy as np


def build_statistics(train_features: np.ndarray, train_labels: np.ndarray, alpha: float = 0.99):
    healthy = train_features[train_labels == 0]
    mu = healthy.mean(axis=0)
    cov = np.cov(healthy.T) + 1e-6 * np.eye(healthy.shape[1])
    inv_cov = np.linalg.inv(cov)

    diff = healthy - mu
    t2_vals = np.einsum("bi,ij,bj->b", diff, inv_cov, diff)
    t2_thr = float(np.quantile(t2_vals, alpha))

    # SPE 通过 PCA 近似
    u, s, vt = np.linalg.svd(healthy - mu, full_matrices=False)
    k = max(1, healthy.shape[1] // 2)
    p = vt[:k].T
    recon = (healthy - mu) @ p @ p.T
    spe = np.sum(((healthy - mu) - recon) ** 2, axis=1)
    spe_thr = float(np.quantile(spe, alpha))

    return {"mu": mu, "inv_cov": inv_cov, "p": p, "t2_thr": t2_thr, "spe_thr": spe_thr}


def detect(stats: dict, features: np.ndarray):
    mu, inv_cov, p = stats["mu"], stats["inv_cov"], stats["p"]
    diff = features - mu
    t2 = np.einsum("bi,ij,bj->b", diff, inv_cov, diff)
    recon = diff @ p @ p.T
    spe = np.sum((diff - recon) ** 2, axis=1)
    alarm = ((t2 > stats["t2_thr"]) | (spe > stats["spe_thr"]))
    return t2, spe, alarm.astype(int)
