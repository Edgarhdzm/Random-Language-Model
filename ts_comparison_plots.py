# ts_comparison_plots.py
import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.serialization import add_safe_globals

# permitir argparse.Namespace en cargas "seguras"
add_safe_globals([argparse.Namespace])

def _to_1d_tensor(x):
    if isinstance(x, torch.Tensor):
        t = x
    elif isinstance(x, np.ndarray):
        t = torch.from_numpy(x)
    else:
        t = torch.as_tensor(np.asarray(x))
    return t.squeeze().to(torch.float32)

def _load_runs(beta, ts, base_path, n_runs=8, filename_pattern="rlm64L1_b{beta}_ts{ts}_{run}.pt"):
    runs = []
    for run in range(1, n_runs + 1):
        fname = filename_pattern.format(beta=beta, ts=ts, run=run)
        fpath = os.path.join(base_path, fname)
        try:
            # con weights_only=True y la allowlist ya añadida
            res = torch.load(fpath, map_location="cpu", weights_only=True)
            dyn = res["output"]["dynamics"]
            t = [d["t"] for d in dyn]
            tl = [d["testloss"] for d in dyn]
            ent = res["output"]["entropy"]
            runs.append({
                "t": _to_1d_tensor(t),
                "testloss": _to_1d_tensor(tl),
                "entropy": float(ent) if not isinstance(ent, (float, int)) else ent
            })
        except FileNotFoundError:
            continue
    return runs

def _aggregate_runs(runs):
    if not runs:
        return None
    L = min(r["t"].numel() for r in runs)
    T = torch.stack([r["t"][:L] for r in runs], dim=0)
    TL = torch.stack([r["testloss"][:L] for r in runs], dim=0)
    ENT = torch.stack([torch.full((L,), float(r["entropy"])) for r in runs], dim=0)
    t_mean = T.mean(dim=0)
    tl_mean = TL.mean(dim=0)
    tl_std = TL.std(dim=0)
    ent_mean = ENT.mean(dim=0)
    diff = TL - ENT
    diff_mean = diff.mean(dim=0)
    diff_std = diff.std(dim=0)
    return {
        "t": t_mean,
        "testloss_mean": tl_mean,
        "testloss_std": tl_std,
        "entropy_mean": ent_mean,
        "diff_mean": diff_mean,
        "diff_std": diff_std
    }

def compare_ts_for_beta(beta_value, ts_values, base_path, n_runs=8, filename_pattern="rlm64L1_b{beta}_ts{ts}_{run}.pt"):
    fig, ax = plt.subplots(1, 2, figsize=(20, 7))
    for ts in ts_values:
        runs = _load_runs(beta_value, ts, base_path, n_runs=n_runs, filename_pattern=filename_pattern)
        stats = _aggregate_runs(runs)
        if stats is None:
            print(f"Sin datos para beta={beta_value}, ts={ts}")
            continue
        t = stats["t"]
        ax[0].plot(t, stats["testloss_mean"], label=f"ts={ts}")
        ax[0].fill_between(t, stats["testloss_mean"] - stats["testloss_std"], stats["testloss_mean"] + stats["testloss_std"], alpha=0.2)
        ax[0].plot(t, stats["entropy_mean"], linestyle="dashed")
        ax[1].plot(t, stats["diff_mean"], label=f"ts={ts}")
        ax[1].fill_between(t, stats["diff_mean"] - stats["diff_std"], stats["diff_mean"] + stats["diff_std"], alpha=0.2)
    ax[0].set_xscale("log")
    ax[1].set_xscale("log")
    ax[1].set_yscale("log")
    ax[0].set_xlabel("t")
    ax[1].set_xlabel("t")
    ax[0].set_ylabel("test loss")
    ax[1].set_ylabel("test loss - entropy")
    ax[0].grid(True, alpha=0.3)
    ax[1].grid(True, alpha=0.3)
    ax[0].legend()
    ax[1].legend()
    fig.suptitle(f"Comparación ts para beta={beta_value}")
    plt.show()
