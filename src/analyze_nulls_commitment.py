#!/usr/bin/env python3
"""Pre-registered commitment null on rat orbitofrontal cortex (Zenodo accession 16997337,
Schiereck et al., Neuron 2026). Tests whether autonomous recurrence tracks behavioural
commitment using two within-session, value-matched contrasts on OFC population trajectories:
  C1: among opt-out trials, high-wait (committed / confident) versus low-wait (gave up),
      split at wait-time tertiles.
  C2: opt-in (committed to wait) versus opt-out (abandoned).
Read-outs per condition (from demix.py): sdmdc_dom (dominant autonomous eigenvalue),
auto_unique (variance-partition recurrence), and r2_input. The prediction under test is that
more commitment yields more autonomous recurrence (higher auto_unique). Each contrast is
value-matched by repeated subsampling and aggregated across sessions (median signed difference
and sign counts; per-session records support a downstream Wilcoxon).

Input: a pickle of per-session OFC trajectories aligned to commitment (env DATA_DIR or argv[1]);
each session provides population trajectories, trial value, opt-out flags, wait times, and a
violation mask. Output: per-session JSON (argv[2]).
"""
import os, sys, json, pickle, warnings
warnings.filterwarnings("ignore")
import numpy as np
from multiprocessing import Pool
from demix import sdmdc_dom, autonomy_index

PKL = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.environ.get("DATA_DIR", "./data"), "ofc_commit_trajectories.pkl")
OUT = sys.argv[2] if len(sys.argv) > 2 else "./results/nulls_commitment.json"
NPROC = int(sys.argv[3]) if len(sys.argv) > 3 else (os.cpu_count() or 1)
NDRAW = 20; MIN_TRIALS = 8; MAXTRIALS = 70; SEED = 20260630; NDELAYS = 3; RANK = 8; TAU = 3.0

def cap(idx, rng):
    return idx if len(idx) <= MAXTRIALS else rng.choice(idx, MAXTRIALS, replace=False)

def onset_kernel(T):
    return np.exp(-np.arange(T)/TAU)

def build_input(v_sub, T):
    k = onset_kernel(T); n = len(v_sub); U = np.zeros((n, T, 2), float)
    for i, vv in enumerate(v_sub):
        U[i, :, 0] = k; U[i, :, 1] = vv * k
    flat = U.reshape(-1, 2); mu = flat.mean(0); sd = flat.std(0); sd[sd < 1e-9] = 1.0
    return (U - mu) / sd

def level_match(a, b, v, rng):
    la = v[a]; lb = v[b]; sa = []; sb = []
    for lv in np.unique(np.concatenate([la, lb])):
        ii = a[la == lv]; cc = b[lb == lv]; m = min(len(ii), len(cc))
        if m == 0:
            continue
        sa.append(rng.choice(ii, m, replace=False)); sb.append(rng.choice(cc, m, replace=False))
    if not sa:
        return None, None
    return np.concatenate(sa), np.concatenate(sb)

def cond_readouts(idx, pct, v):
    X = np.stack([pct[i].T for i in idx], axis=0)
    U = build_input(v[idx], X.shape[1]); ai = autonomy_index(X, U)
    return dict(sdmdc_dom=sdmdc_dom(X, U, rank=RANK, nd=NDELAYS),
                auto_unique=ai["auto_unique"], r2_input=ai["r2_input"])

def contrast(hi, lo, pct, v, rng, keys):
    acc = {k: {"h": [], "l": []} for k in keys}
    for _ in range(NDRAW):
        sh, sl = level_match(hi, lo, v, rng)
        if sh is None or len(sh) < MIN_TRIALS:
            continue
        rh = cond_readouts(sh, pct, v); rl = cond_readouts(sl, pct, v)
        for k in keys:
            if np.isfinite(rh[k]) and np.isfinite(rl[k]):
                acc[k]["h"].append(rh[k]); acc[k]["l"].append(rl[k])
    out = {}
    for k in keys:
        h = np.array(acc[k]["h"]); l = np.array(acc[k]["l"])
        if h.size >= 5:
            out[f"{k}_hi"] = float(np.mean(h)); out[f"{k}_lo"] = float(np.mean(l)); out[f"d_{k}"] = float(np.mean(h)-np.mean(l))
    return out if "d_auto_unique" in out else None

def work(item):
    fn, S = item
    try:
        pct = S["pctraj"].astype(np.float64); vios = np.asarray(S["vios"]).astype(bool)
        v = np.asarray(S["v"]).astype(float); optout = np.asarray(S["optout"]).astype(float)
        wait = np.asarray(S["wait_time"]).astype(float); nonvio = ~vios
        rng = np.random.default_rng(abs(hash(fn)) % (2**31) + SEED)
        keys = ["sdmdc_dom", "auto_unique", "r2_input"]
        rec = dict(file=fn, rat=S.get("rat"))
        # C1: wait-time tertiles among opt-out trials
        oo = np.where(nonvio & (optout == 1) & np.isfinite(wait))[0]
        if oo.size >= 3*MIN_TRIALS:
            w = wait[oo]; p33, p67 = np.percentile(w, [33, 67])
            hi = oo[w >= p67]; lo = oo[w <= p33]
            if hi.size >= MIN_TRIALS and lo.size >= MIN_TRIALS:
                c1 = contrast(cap(hi, rng), cap(lo, rng), pct, v, rng, keys)
                if c1:
                    rec["C1"] = c1; rec["C1_n_hi"] = int(hi.size); rec["C1_n_lo"] = int(lo.size)
        # C2: opt-in vs opt-out
        oi = np.where(nonvio & (optout == 0))[0]; oot = np.where(nonvio & (optout == 1))[0]
        if oi.size >= MIN_TRIALS and oot.size >= MIN_TRIALS:
            c2 = contrast(cap(oi, rng), cap(oot, rng), pct, v, rng, keys)
            if c2:
                rec["C2"] = c2; rec["C2_n_in"] = int(oi.size); rec["C2_n_out"] = int(oot.size)
        return rec if ("C1" in rec or "C2" in rec) else None
    except Exception as e:
        return dict(file=fn, error=repr(e)[:160])

def main():
    data = pickle.load(open(PKL, "rb")); items = list(data.items())
    print(f"{len(items)} sessions; commitment contrasts; value-matched {NDRAW} draws", flush=True)
    with Pool(NPROC) as pl:
        res = [r for r in pl.map(work, items) if r is not None and "error" not in r]
    json.dump(res, open(OUT, "w"), indent=1, default=str)
    for C in ("C1", "C2"):
        sub = [r[C] for r in res if C in r]
        print(f"\n{C}: n={len(sub)} sessions", flush=True)
        for k in ("auto_unique", "sdmdc_dom", "r2_input"):
            dd = np.array([s[f"d_{k}"] for s in sub if f"d_{k}" in s])
            if dd.size:
                print(f"  {k}: d_med={np.median(dd):+.4f} pos={int((dd>0).sum())}/{dd.size}", flush=True)
    print("wrote", OUT)

if __name__ == "__main__":
    main()
