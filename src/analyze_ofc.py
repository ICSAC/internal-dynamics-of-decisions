#!/usr/bin/env python3
"""Recurrent-versus-input demix of rat orbitofrontal cortex population trajectories.

When the known task input (cue onset plus offered value, time-locked) is regressed out, does the
autonomous (recurrent) component still rise under inference (early trials after a block switch,
trial-since-switch < 10) relative to a value-matched control (trial-since-switch >= 20)? A rise
indicates that the dominant-eigenvalue increase reflects genuine autonomous recurrence rather than
input drive.

Input: the per-session PC-trajectory pickle produced by extract_ofc.py (argv[1], default
./data/ofc_pctraj.pkl), built from the rat OFC value-based decision dataset (Zenodo accession
16997337; Schiereck et al., Neuron 2026).

Per condition, averaged over NDRAW value-matched subsamples, four read-outs are computed:
  sdmdc_dom   = dominant |eigenvalue| of the latent autonomous operator A_v from SubspaceDMDc
                (input-cleaned autonomous recurrence)
  plain_dom   = dominant |eigenvalue| of plain DMD with no control input (input-contaminated)
  auto_unique = R2(x'|x,u) - R2(x'|u)  (recurrence variance beyond the known input)
  r2_input    = R2(x'|u)               (how input-driven the trajectory is)

The exogenous input u_t (T bins, non-circular, task-defined) has two channels
[onset_kernel, value*onset_kernel] with onset_kernel = exp(-t/TAU) from window start and value the
offered reward level (1-5); channels are z-scored. Value-matched subsampling equalizes the value
composition across the inference and control conditions, so any autonomous difference is not an
input difference.

Output: a JSON list (argv[2], default ./results/inputdsa_brain_ofc.json) of per-session records
with the inference, control, and difference values for each read-out; a label-permutation null and
paired across-session statistics are computed separately from this JSON.
"""
import sys, os, json, pickle, warnings
warnings.filterwarnings("ignore")
import numpy as np
from multiprocessing import Pool
from demix import sdmdc_dom, autonomy_index

PKL = sys.argv[1] if len(sys.argv) > 1 else "./data/ofc_pctraj.pkl"
OUT = sys.argv[2] if len(sys.argv) > 2 else "./results/inputdsa_brain_ofc.json"
NPROC = int(sys.argv[3]) if len(sys.argv) > 3 else (os.cpu_count() or 4)
NDRAW = 20; MIN_TRIALS = 8; SEED = 20260630; NDELAYS = 3; RANK = 8; TAU = 3.0

def _to_np(M):
    return np.asarray(M.cpu() if hasattr(M, "cpu") else M)

def onset_kernel(T):
    k = np.exp(-np.arange(T)/TAU)
    return k

def build_input(v_sub, T):
    """U: (n, T, 2) exogenous: [onset, value*onset]; channels z-scored across all (trial,t)."""
    k = onset_kernel(T)
    n = len(v_sub)
    U = np.zeros((n, T, 2), float)
    for i, v in enumerate(v_sub):
        U[i, :, 0] = k
        U[i, :, 1] = v * k
    # z-score each channel across all samples
    flat = U.reshape(-1, 2)
    mu = flat.mean(0); sd = flat.std(0); sd[sd < 1e-9] = 1.0
    U = (U - mu) / sd
    return U

def plain_dmd_dom(X, nd=NDELAYS):
    from DSA import DMD
    try:
        d = DMD(X, n_delays=nd, rank=X.shape[2]*nd, device="cpu", verbose=False); d.fit()
        return float(np.max(np.abs(np.linalg.eigvals(_to_np(d.A_v)))))
    except Exception:
        return np.nan

def level_match(inf, ctl, v, rng):
    li = v[inf]; lc = v[ctl]; si = []; sc = []
    for lv in np.unique(np.concatenate([li, lc])):
        ii = inf[li == lv]; cc = ctl[lc == lv]; m = min(len(ii), len(cc))
        if m == 0:
            continue
        si.append(rng.choice(ii, m, replace=False)); sc.append(rng.choice(cc, m, replace=False))
    if not si:
        return None, None
    return np.concatenate(si), np.concatenate(sc)

def cond_readouts(idx, pct, v):
    """X: (n,T,R) from pct[idx]; U from v[idx]. returns dict of readouts."""
    X = np.stack([pct[i].T for i in idx], axis=0)   # n x T x R
    T = X.shape[1]
    U = build_input(v[idx], T)
    ai = autonomy_index(X, U)
    return dict(sdmdc_dom=sdmdc_dom(X, U, rank=RANK, nd=NDELAYS),
                plain_dom=plain_dmd_dom(X),
                auto_unique=ai["auto_unique"], r2_input=ai["r2_input"],
                r2_full=ai["r2_full"], autonomy_index=ai["autonomy_index"])

def work(item):
    fn, S = item
    try:
        pct = S["pctraj"].astype(np.float64)        # trials x R x bins
        tss = np.asarray(S["tss"]); vios = np.asarray(S["vios"]).astype(bool); v = np.asarray(S["v"]).astype(float)
        nonvio = ~vios
        inf = np.where(nonvio & (tss < 10))[0]; ctl = np.where(nonvio & (tss >= 20))[0]
        if inf.size < MIN_TRIALS or ctl.size < MIN_TRIALS:
            return None
        rng = np.random.default_rng(abs(hash(fn)) % (2**31) + SEED)
        keys = ["sdmdc_dom", "plain_dom", "auto_unique", "r2_input", "r2_full", "autonomy_index"]
        acc = {k: {"i": [], "c": []} for k in keys}
        nmatch = []
        for _ in range(NDRAW):
            si, sc = level_match(inf, ctl, v, rng)
            if si is None or len(si) < MIN_TRIALS:
                continue
            ri = cond_readouts(si, pct, v); rc = cond_readouts(sc, pct, v)
            for k in keys:
                if np.isfinite(ri[k]) and np.isfinite(rc[k]):
                    acc[k]["i"].append(ri[k]); acc[k]["c"].append(rc[k])
            nmatch.append(len(si))
        if len(acc["sdmdc_dom"]["i"]) < 5:
            return None
        rec = dict(file=fn, rat=S.get("rat"), n_units=int(S.get("n_units", 0)),
                   n_inf=int(inf.size), n_ctl=int(ctl.size), n_matched=float(np.mean(nmatch)))
        for k in keys:
            i = np.array(acc[k]["i"]); c = np.array(acc[k]["c"])
            if i.size >= 5:
                rec[f"{k}_inf"] = float(np.mean(i)); rec[f"{k}_ctl"] = float(np.mean(c))
                rec[f"d_{k}"] = float(np.mean(i) - np.mean(c))
        return rec
    except Exception as e:
        return dict(file=fn, error=repr(e)[:160])

def perm_work(item):
    """one label-permutation draw for the headline sdmdc_dom + auto_unique aggregate null."""
    fn, S, seed = item
    try:
        pct = S["pctraj"].astype(np.float64)
        tss = np.asarray(S["tss"]); vios = np.asarray(S["vios"]).astype(bool); v = np.asarray(S["v"]).astype(float)
        nonvio = ~vios
        inf = np.where(nonvio & (tss < 10))[0]; ctl = np.where(nonvio & (tss >= 20))[0]
        if inf.size < MIN_TRIALS or ctl.size < MIN_TRIALS:
            return None
        rng = np.random.default_rng(seed)
        pool = np.concatenate([inf, ctl]); nI = inf.size
        perm = rng.permutation(pool); si = perm[:nI]; sc = perm[nI:]
        # value-match the permuted labels too
        sm, scm = level_match(si, sc, v, rng)
        if sm is None or len(sm) < MIN_TRIALS:
            return None
        ri = cond_readouts(sm, pct, v); rc = cond_readouts(scm, pct, v)
        return dict(d_sdmdc=ri["sdmdc_dom"]-rc["sdmdc_dom"] if np.isfinite(ri["sdmdc_dom"]) and np.isfinite(rc["sdmdc_dom"]) else np.nan,
                    d_auto=ri["auto_unique"]-rc["auto_unique"] if np.isfinite(ri["auto_unique"]) and np.isfinite(rc["auto_unique"]) else np.nan)
    except Exception:
        return None

def main():
    data = pickle.load(open(PKL, "rb")); items = list(data.items())
    print(f"{len(items)} sessions; value-matched {NDRAW} draws; SubspaceDMDc nd={NDELAYS} rank={RANK}; Pool({NPROC})", flush=True)
    with Pool(NPROC) as pl:
        res = [r for r in pl.map(work, items) if r is not None and "error" not in r]
    os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=1, default=str)
    print(f"usable sessions: {len(res)}", flush=True)
    # aggregate statistics (paired Wilcoxon across sessions) are computed separately from the
    # JSON; quick medians are printed here as a sanity check.
    for k in ("sdmdc_dom", "plain_dom", "auto_unique", "r2_input"):
        dd = np.array([r[f"d_{k}"] for r in res if f"d_{k}" in r])
        di = np.array([r[f"{k}_inf"] for r in res if f"{k}_inf" in r])
        dc = np.array([r[f"{k}_ctl"] for r in res if f"{k}_ctl" in r])
        if dd.size:
            print(f"  {k}: inf_med={np.median(di):.4f} ctl_med={np.median(dc):.4f} "
                  f"d_med={np.median(dd):+.4f} pos={int((dd>0).sum())} neg={int((dd<0).sum())} n={dd.size}", flush=True)
    print(f"wrote {OUT}")

if __name__ == "__main__":
    main()
