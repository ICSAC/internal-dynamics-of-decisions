#!/usr/bin/env python3
"""Recurrent-versus-input demix of primate entorhinal cortex during mental navigation
(DANDI:000897, Neupane et al. 2024), the second cortical anchor. Inference trials (MNAV,
trial_type 3, where the map is occluded so the traversal must be mentally constructed) are
compared against visible-map control trials (NTS), recorded from the same simultaneously active
units, with trials step-matched across conditions. The exogenous input
u_t = [onset_kernel, steps * onset_kernel] models the movement-onset-evoked, step-graded drive.
For each session, and for each of NDRAW step-matched bootstrap draws, the script computes the
SubspaceDMDc dominant autonomous eigenvalue, a plain time-delay DMD eigenvalue, and the
variance-partition autonomy read-outs (auto_unique, r2_input, r2_full) under inference and under
control, then averages over draws.

Input: a pickle of per-session PC trajectories from extract_entorhinal.py (path from argv[1],
default DATA_DIR/entorhinal_trajectories.pkl). Output: a JSON list (path from argv[2], default
./results/entorhinal.json) of per-session inference/control means and their differences for each
read-out. Optional argv[3] sets the worker-pool size.
"""
import os, sys, json, pickle, warnings
warnings.filterwarnings("ignore")
import numpy as np
from multiprocessing import Pool
from demix import sdmdc_dom, autonomy_index

PKL = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.environ.get("DATA_DIR", "./data"), "entorhinal_trajectories.pkl")
OUT = sys.argv[2] if len(sys.argv) > 2 else "./results/entorhinal.json"
NPROC = int(sys.argv[3]) if len(sys.argv) > 3 else (os.cpu_count() or 4)
NDRAW = 20; MIN_TRIALS = 8; SEED = 20260630; NDELAYS = 3; RANK = 8; TAU = 3.0

def _to_np(M):
    return np.asarray(M.cpu() if hasattr(M, "cpu") else M)

def onset_kernel(T):
    return np.exp(-np.arange(T)/TAU)

def build_input(steps_sub, T):
    k = onset_kernel(T); n = len(steps_sub)
    U = np.zeros((n, T, 2), float)
    for i, s in enumerate(steps_sub):
        U[i, :, 0] = k; U[i, :, 1] = s * k
    flat = U.reshape(-1, 2); mu = flat.mean(0); sd = flat.std(0); sd[sd < 1e-9] = 1.0
    return (U - mu) / sd

def plain_dmd_dom(X, nd=NDELAYS):
    from DSA import DMD
    try:
        d = DMD(X, n_delays=nd, rank=X.shape[2]*nd, device="cpu", verbose=False); d.fit()
        return float(np.max(np.abs(np.linalg.eigvals(_to_np(d.A_v)))))
    except Exception:
        return np.nan

def step_match(inf, ctl, steps, rng):
    li = steps[inf]; lc = steps[ctl]; si = []; sc = []
    for lv in np.unique(np.concatenate([li, lc])):
        ii = inf[li == lv]; cc = ctl[lc == lv]; m = min(len(ii), len(cc))
        if m == 0:
            continue
        si.append(rng.choice(ii, m, replace=False)); sc.append(rng.choice(cc, m, replace=False))
    if not si:
        return None, None
    return np.concatenate(si), np.concatenate(sc)

def cond_readouts(idx, pct, steps):
    X = np.stack([pct[i].T for i in idx], axis=0)   # n x T x R
    U = build_input(steps[idx], X.shape[1])
    ai = autonomy_index(X, U)
    return dict(sdmdc_dom=sdmdc_dom(X, U, rank=RANK, nd=NDELAYS), plain_dom=plain_dmd_dom(X),
                auto_unique=ai["auto_unique"], r2_input=ai["r2_input"], r2_full=ai["r2_full"])

def work(item):
    fn, S = item
    try:
        pct = S["pctraj"].astype(np.float64); tt = np.asarray(S["tt"]); steps = np.asarray(S["steps"]).astype(float)
        ctt = int(S["ctt"])
        inf = np.where(tt == 3)[0]; ctl = np.where(tt == ctt)[0]
        if inf.size < MIN_TRIALS or ctl.size < MIN_TRIALS:
            return None
        rng = np.random.default_rng(abs(hash(fn)) % (2**31) + SEED)
        keys = ["sdmdc_dom", "plain_dom", "auto_unique", "r2_input", "r2_full"]
        acc = {k: {"i": [], "c": []} for k in keys}; nmatch = []
        for _ in range(NDRAW):
            si, sc = step_match(inf, ctl, steps, rng)
            if si is None or len(si) < MIN_TRIALS:
                continue
            ri = cond_readouts(si, pct, steps); rc = cond_readouts(sc, pct, steps)
            for k in keys:
                if np.isfinite(ri[k]) and np.isfinite(rc[k]):
                    acc[k]["i"].append(ri[k]); acc[k]["c"].append(rc[k])
            nmatch.append(len(si))
        if len(acc["auto_unique"]["i"]) < 5:
            return None
        rec = dict(file=fn, subj=S.get("subj"), n_units=int(S.get("n_units", 0)),
                   n_inf=int(inf.size), n_ctl=int(ctl.size), n_matched=float(np.mean(nmatch)))
        for k in keys:
            i = np.array(acc[k]["i"]); c = np.array(acc[k]["c"])
            if i.size >= 5:
                rec[f"{k}_inf"] = float(np.mean(i)); rec[f"{k}_ctl"] = float(np.mean(c)); rec[f"d_{k}"] = float(np.mean(i)-np.mean(c))
        return rec
    except Exception as e:
        return dict(file=fn, error=repr(e)[:160])

def main():
    data = pickle.load(open(PKL, "rb")); items = list(data.items())
    print(f"{len(items)} sessions; step-matched {NDRAW} draws; SubspaceDMDc nd={NDELAYS} rank={RANK}", flush=True)
    with Pool(NPROC) as pl:
        res = [r for r in pl.map(work, items) if r is not None and "error" not in r]
    os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=1, default=str)
    print(f"usable sessions: {len(res)}", flush=True)
    for k in ("sdmdc_dom", "plain_dom", "auto_unique", "r2_input"):
        dd = np.array([r[f"d_{k}"] for r in res if f"d_{k}" in r])
        di = np.array([r[f"{k}_inf"] for r in res if f"{k}_inf" in r]); dc = np.array([r[f"{k}_ctl"] for r in res if f"{k}_ctl" in r])
        if dd.size:
            print(f"  {k}: inf_med={np.median(di):.4f} ctl_med={np.median(dc):.4f} d_med={np.median(dd):+.4f} pos={int((dd>0).sum())} neg={int((dd<0).sum())} n={dd.size}", flush=True)
    print("wrote", OUT)

if __name__ == "__main__":
    main()
