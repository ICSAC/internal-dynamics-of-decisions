#!/usr/bin/env python3
"""Train networks to matched accuracy and save R-PC trajectories + per-bin decision probs.

Frozen per PREREG-CMC.md sec.2-4. Per-bin cross-entropy against the trial label (identical for all
archs); reported accuracy is FINAL-BIN test accuracy. Penultimate/hidden trajectories are reduced to
R=8 PCs (PCA fit on pooled test hidden) and saved. This step touches NO dynamical measure.

Usage: python train.py <task>            # loops all 6 archs x 6 seeds
       python train.py <task> <arch> <seed>
"""
import os, sys, json, time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
import task as TASK
from models import make, FAMILY, ARCHS

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
R = 8
SEEDS = [0, 1, 2, 3, 4, 5]
EPOCHS, BATCH, LR, WD, PATIENCE = 60, 256, 1e-3, 1e-4, 8
OUT = os.environ.get("CMC_OUT", "./artifacts")


def set_seed(s):
    np.random.seed(s); torch.manual_seed(s)
    if DEVICE == "cuda":
        torch.cuda.manual_seed_all(s)


def get_data(task):
    Utr, ytr, _ = TASK.gen_trials(6000, task, TASK.DATA_SEED)
    Uva, yva, _ = TASK.gen_trials(2000, task, TASK.DATA_SEED + 1)
    Ute, yte, gte = TASK.gen_trials(4000, task, TASK.DATA_SEED + 2)
    Utr_s, mu, sd = TASK.standardize(Utr)
    Uva_s, _, _ = TASK.standardize(Uva, mu, sd)
    Ute_s, _, _ = TASK.standardize(Ute, mu, sd)
    return (Utr_s, ytr), (Uva_s, yva), (Ute_s, yte, gte, Ute), (mu, sd)


def to_t(x):
    return torch.tensor(x, dtype=torch.float32, device=DEVICE)


def final_acc(model, U, y):
    model.eval()
    with torch.no_grad():
        out = []
        for i in range(0, len(U), 1024):
            _, lg = model(to_t(U[i:i+1024]))
            out.append(lg[:, -1, :].argmax(1).cpu().numpy())
        pred = np.concatenate(out)
    return float((pred == y).mean())


def train_one(task, arch, seed, data):
    (Utr, ytr), (Uva, yva), (Ute, yte, gte, Ute_raw), _ = data
    set_seed(seed)
    model, width, nparams = make(arch)
    model = model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    Utr_t = to_t(Utr); ytr_t = torch.tensor(ytr, device=DEVICE)
    n = len(Utr); best_va, best_state, bad = -1.0, None, 0
    for ep in range(EPOCHS):
        model.train(); perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n, BATCH):
            idx = perm[i:i+BATCH]
            u = Utr_t[idx]; yb = ytr_t[idx]
            _, lg = model(u)                                  # (B,T,2)
            B, Tt, _ = lg.shape
            loss = F.cross_entropy(lg.reshape(B * Tt, 2),
                                   yb[:, None].expand(B, Tt).reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        va = final_acc(model, Uva, yva)
        if va > best_va:
            best_va, best_state, bad = va, {k: v.detach().clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    model.load_state_dict(best_state)
    te = final_acc(model, Ute, yte)

    # extract penultimate hidden on test, PCA -> R, per-bin P(right)
    model.eval(); Hs, Ps = [], []
    with torch.no_grad():
        for i in range(0, len(Ute), 1024):
            H, lg = model(to_t(Ute[i:i+1024]))
            Hs.append(H.cpu().numpy()); Ps.append(F.softmax(lg, -1)[:, :, 1].cpu().numpy())
    Hs = np.concatenate(Hs).astype(np.float64)        # (N,T,hidden)
    Ps = np.concatenate(Ps).astype(np.float64)        # (N,T)
    N, Tt, hid = Hs.shape
    flat = Hs.reshape(N * Tt, hid)
    mu = flat.mean(0); Uc = flat - mu
    # top-R PCs via SVD
    _, _, Vt = np.linalg.svd(Uc, full_matrices=False)
    basis = Vt[:R].T                                  # (hid,R)
    X = ((Hs.reshape(N * Tt, hid) - mu) @ basis).reshape(N, Tt, R)
    return dict(arch=arch, family=FAMILY[arch], seed=seed, task=task, width=width,
                nparams=nparams, test_acc=te, val_acc=best_va, X=X, P=Ps, hidden=hid)


def main():
    task = sys.argv[1] if len(sys.argv) > 1 else "primary"
    os.makedirs(OUT, exist_ok=True)
    data = get_data(task)
    # save shared test U_raw, y, ideal-observer commit once per task
    (_, _), (_, _), (Ute_s, yte, gte, Ute_raw), _ = data
    tc_io = TASK.ideal_observer_commit(Ute_raw, task)
    np.savez_compressed(os.path.join(OUT, f"shared_{task}.npz"),
                        U=Ute_raw.astype(np.float64), y=yte, gamma=gte, tc_io=tc_io)
    if len(sys.argv) >= 4:
        archs, seeds = [sys.argv[2]], [int(sys.argv[3])]
    else:
        archs, seeds = ARCHS, SEEDS
    table = []
    for arch in archs:
        for seed in seeds:
            t0 = time.time()
            r = train_one(task, arch, seed, data)
            np.savez_compressed(os.path.join(OUT, f"net_{task}_{arch}_{seed}.npz"),
                                X=r["X"], P=r["P"])
            row = {k: r[k] for k in ("arch", "family", "seed", "task", "width",
                                     "nparams", "test_acc", "val_acc", "hidden")}
            row["sec"] = round(time.time() - t0, 1)
            table.append(row)
            print(f"{task} {arch:4s} s{seed} acc={r['test_acc']:.4f} "
                  f"params={r['nparams']:,} ({row['sec']}s)", flush=True)
    json.dump(table, open(os.path.join(OUT, f"acc_table_{task}.json"), "w"), indent=1)
    accs = [r["test_acc"] for r in table]
    print(f"\n{task}: acc range [{min(accs):.4f},{max(accs):.4f}] mean={np.mean(accs):.4f}", flush=True)


if __name__ == "__main__":
    main()
