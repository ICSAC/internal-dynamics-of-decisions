#!/usr/bin/env python3
"""Time-resolved autonomous-versus-input dynamics around decision commitment in
perceptual-decision Neuropixels data (Dryad doi:10.5061/dryad.sj3tx96dm and
github.com/Brody-Lab/decision_dynamics_commitment, Luo et al., Nature 2025).

For each session, a canonical SubspaceDMDc (N4SID) model is fit in sliding windows over the
population trajectory, separately aligned to the moment of commitment and to stimulus onset.
Two read-outs are reported per window:
  dom          = magnitude of the dominant eigenvalue of the latent autonomous operator A_v,
                 i.e. autonomous-recurrence strength;
  flow_balance = mean ||A_v z||^2 / (||A_v z||^2 + ||B_v u||^2) in the N4SID latent space,
                 i.e. the autonomous (intrinsic) fraction of the latent flow; a rise indicates
                 the population locking into an autonomous attractor.
Window read-outs are aggregated across sessions to give mean +/- s.e.m. curves for the
commitment-aligned and stimulus-aligned conditions.

Input: a pickle of per-session aligned trajectories (env DATA_DIR or argv[1]); each session
provides commitment- and stimulus-aligned trajectories, the aligned exogenous input, the window
offsets, and (for commitment alignment) per-trial movement times used to drop post-movement data.
Output: per-session JSON (argv[2]) plus a "<out>_summary.json" with the aggregate curves.
"""
import os, sys, json, pickle, warnings
warnings.filterwarnings("ignore")
import numpy as np
from multiprocessing import Pool
# subspace state-space identification (N4SID / SubspaceDMDc); see requirements.txt
from DSA import SubspaceDMDc

PKL = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.environ.get("DATA_DIR", "./data"), "commitment_trajectories.pkl")
OUT = sys.argv[2] if len(sys.argv) > 2 else "./results/commitment.json"
NPROC = int(sys.argv[3]) if len(sys.argv) > 3 else (os.cpu_count() or 1)
W = 8; RANK = 8; NDELAYS = 2; MIN_TR = 20

def to_np(M):
    return np.asarray(M.cpu() if hasattr(M, "cpu") else M)

def flow_readouts(X, U):
    """SubspaceDMDc -> (dom|lambda|(A_v), autonomous flow fraction)."""
    try:
        d = SubspaceDMDc(X, control_data=U, n_delays=NDELAYS, rank=RANK, device="cpu", verbose=False)
        d.fit()
        Av = to_np(d.A_v); Bv = to_np(d.B_v)
        dom = float(np.max(np.abs(np.linalg.eigvals(Av))))
        z, znext, u = d.get_training_latent_states(return_aligned=True)
        z = to_np(z); u = to_np(u); m = min(Bv.shape[1], u.shape[1])
        auto = float(np.mean(np.sum((z @ Av.T)**2, axis=1)))
        inp = float(np.mean(np.sum((u[:, :m] @ Bv[:, :m].T)**2, axis=1)))
        bal = auto / (auto + inp + 1e-12)
        if not (np.isfinite(dom) and np.isfinite(bal)):
            return None
        return dom, bal
    except Exception:
        return None

def session_curve(traj, ev, offs, mv=None, mask_move=False):
    ntr, nb, R = traj.shape
    dt = float(offs[1] - offs[0])
    rows = []
    for ws in range(nb - W + 1):
        sel = np.arange(ntr)
        if mask_move and mv is not None:
            wend = offs[ws] + W * dt
            sel = np.where(wend <= mv)[0]
        if len(sel) < MIN_TR:
            rows.append(None); continue
        r = flow_readouts(traj[sel, ws:ws+W, :], ev[sel, ws:ws+W, :])
        if r is None:
            rows.append(None); continue
        rows.append(dict(center=float(offs[ws] + W/2*dt), dom=r[0], flow_balance=r[1], n=len(sel)))
    return rows

def work(item):
    k, v = item
    try:
        cc = session_curve(v["commit_traj"].astype(np.float64), v["ev_commit"].astype(np.float64),
                           v["commit_off"], v["mv_rel_commit"], mask_move=True)
        sc = session_curve(v["stim_traj"].astype(np.float64), v["ev_stim"].astype(np.float64),
                           v["stim_off"], mask_move=False)
        return (k, {"commit": cc, "stim": sc})
    except Exception as e:
        return (k, {"error": repr(e)[:120]})

def main():
    data = pickle.load(open(PKL, "rb"))
    items = [(k, v) for k, v in data.items() if "error" not in v and v.get("n_trials", 0) >= MIN_TR]
    print(f"{len(items)} sessions; SubspaceDMDc flow, W={W} rank={RANK} nd={NDELAYS}; Pool({NPROC})", flush=True)
    with Pool(NPROC) as pl:
        res = dict(r for r in pl.map(work, items) if r is not None)
    good = {k: v for k, v in res.items() if "error" not in v}
    json.dump(good, open(OUT, "w"), indent=1, default=str)
    print(f"usable sessions: {len(good)}", flush=True)

    def mean_curve(key_align, key):
        nwin = max((len(v[key_align]) for v in good.values()), default=0)
        out = []
        for i in range(nwin):
            vals = [v[key_align][i][key] for v in good.values() if i < len(v[key_align]) and v[key_align][i]]
            cen = [v[key_align][i]["center"] for v in good.values() if i < len(v[key_align]) and v[key_align][i]]
            if len(vals) >= 10:
                out.append(dict(center=round(float(np.median(cen)), 3), mean=round(float(np.mean(vals)), 4),
                                sem=round(float(np.std(vals)/np.sqrt(len(vals))), 4), n=len(vals)))
        return out
    summ = {"n": len(good),
            "commit_dom": mean_curve("commit", "dom"), "commit_flow": mean_curve("commit", "flow_balance"),
            "stim_dom": mean_curve("stim", "dom"), "stim_flow": mean_curve("stim", "flow_balance")}
    json.dump(summ, open(OUT.replace(".json", "_summary.json"), "w"), indent=1, default=str)
    print("\n=== COMMITMENT-ALIGNED flow_balance (autonomous fraction) ===", flush=True)
    for r in summ["commit_flow"]:
        print(f"  t={r['center']:+.3f}s: flow_balance={r['mean']:.4f}±{r['sem']:.4f}  dom@same n={r['n']}", flush=True)
    print("=== COMMITMENT-ALIGNED dom|lambda| ===", flush=True)
    for r in summ["commit_dom"]:
        print(f"  t={r['center']:+.3f}s: dom={r['mean']:.4f}±{r['sem']:.4f}", flush=True)
    print("=== STIMULUS-ALIGNED flow_balance (control) ===", flush=True)
    for r in summ["stim_flow"]:
        print(f"  t={r['center']:+.3f}s: flow_balance={r['mean']:.4f}±{r['sem']:.4f}", flush=True)
    print("wrote", OUT, flush=True)

if __name__ == "__main__":
    main()
