#!/usr/bin/env python3
"""Variance-partition view of the autonomous-versus-input dynamics around decision commitment in
perceptual-decision Neuropixels data (Dryad doi:10.5061/dryad.sj3tx96dm and
github.com/Brody-Lab/decision_dynamics_commitment, Luo et al., Nature 2025).

For each session, the variance-partition demix is computed in sliding windows over the population
trajectory, separately aligned to commitment and to stimulus onset. Per-window read-outs are
r2_input (next-state variance explained by the known input), auto_unique (recurrence contribution
beyond the input), and dom (magnitude of the dominant eigenvalue of a ridge-regularised
delay-embedded DMD operator). Window read-outs are aggregated across sessions into mean +/- s.e.m.
curves, and a paired per-session locking test asks whether the input-to-autonomous transition is
more sharply locked to the moment of commitment than to stimulus onset (swing = max-min of
r2_input per session; drop = pre minus post r2_input across the alignment point; Wilcoxon).

Input: a pickle of per-session aligned trajectories (env DATA_DIR or argv[1]); each session
provides commitment- and stimulus-aligned trajectories, the aligned exogenous input, the window
offsets, and (for commitment alignment) per-trial movement times used to drop post-movement data.
Output: JSON (argv[2]) with the aggregate curves and the locking statistics.

The variance-partition autonomy index is the pure-NumPy read-out from demix.py.
"""
import os, sys, json, pickle, warnings
warnings.filterwarnings("ignore")
import numpy as np
from scipy import stats
from demix import autonomy_index

PKL = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.environ.get("DATA_DIR", "./data"), "commitment_trajectories.pkl")
OUT = sys.argv[2] if len(sys.argv) > 2 else "./results/commitment_varpart.json"
W = 6; MIN_TR = 20

def dmd_dom(Xw, nd=3, ridge=1e-2):
    P1, P2 = [], []
    for Z in Xw:
        T, R = Z.shape
        if T < nd + 2:
            continue
        H = np.concatenate([Z[nd-1-k:T-k] for k in range(nd)], axis=1)
        P1.append(H[:-1]); P2.append(H[1:])
    if not P1:
        return np.nan
    X1 = np.concatenate(P1, 0); X2 = np.concatenate(P2, 0); d = X1.shape[1]
    A = np.linalg.solve(X1.T @ X1 + ridge*np.eye(d), X1.T @ X2).T
    return float(np.max(np.abs(np.linalg.eigvals(A))))

def session_curve(traj, ev, offs, mv=None, mask_move=False):
    """time-resolved r2_input, auto_unique, dom over sliding windows; returns per-window dict."""
    ntr, nb, R = traj.shape
    nwin = nb - W + 1
    rows = []
    for ws in range(nwin):
        we = ws + W
        sel = np.arange(ntr)
        if mask_move and mv is not None:
            # keep trials whose window end is before movement (relative to commitment)
            wend_t = offs[ws] + W * (offs[1] - offs[0])
            sel = np.where(wend_t <= mv)[0]
        if len(sel) < MIN_TR:
            rows.append(None); continue
        X = traj[sel, ws:we, :]; U = ev[sel, ws:we, :]
        ai = autonomy_index(X, U)
        rows.append(dict(center=float(offs[ws] + W/2*(offs[1]-offs[0])),
                         r2_input=ai["r2_input"], auto_unique=ai["auto_unique"], dom=dmd_dom(X), n=len(sel)))
    return rows

def main():
    data = pickle.load(open(PKL, "rb"))
    sessions = [(k, v) for k, v in data.items() if "error" not in v and v.get("n_trials", 0) >= MIN_TR]
    print(f"{len(sessions)} usable sessions", flush=True)
    # accumulate per-window across sessions
    commit_curves = []; stim_curves = []
    swing_c, swing_s = [], []      # per-session r2_input swing (max-min) each alignment
    drop_c, drop_s = [], []        # per-session r2_input drop (pre minus post window)
    for k, v in sessions:
        co = v["commit_off"]; so = v["stim_off"]
        cc = session_curve(v["commit_traj"], v["ev_commit"], co, v["mv_rel_commit"], mask_move=True)
        sc = session_curve(v["stim_traj"], v["ev_stim"], so, mask_move=False)
        commit_curves.append(cc); stim_curves.append(sc)
        cr = np.array([r["r2_input"] for r in cc if r])
        sr = np.array([r["r2_input"] for r in sc if r])
        if cr.size >= 4:
            swing_c.append(cr.max() - cr.min())
        if sr.size >= 4:
            swing_s.append(sr.max() - sr.min())
        # drop: r2_input in window centered ~-150ms (pre) minus ~+150ms (post), commitment-aligned
        def at(curve, t):
            best = None
            for r in curve:
                if r and (best is None or abs(r["center"]-t) < abs(best["center"]-t)):
                    best = r
            return best
        pc, poc = at(cc, -0.15), at(cc, 0.15)
        if pc and poc:
            drop_c.append(pc["r2_input"] - poc["r2_input"])
        ps, pos = at(sc, 0.15), at(sc, 0.45)   # stimulus-aligned analogous pre/post (accum then late)
        if ps and pos:
            drop_s.append(ps["r2_input"] - pos["r2_input"])

    # aggregate time-resolved mean curves (by window index, since offsets aligned across sessions)
    def mean_curve(curves, key):
        nwin = max(len(c) for c in curves)
        out = []
        for i in range(nwin):
            vals = [c[i][key] for c in curves if i < len(c) and c[i]]
            cen = [c[i]["center"] for c in curves if i < len(c) and c[i]]
            if len(vals) >= 10:
                out.append(dict(center=round(float(np.median(cen)), 3), mean=round(float(np.mean(vals)), 4),
                                sem=round(float(np.std(vals)/np.sqrt(len(vals))), 4), n_sess=len(vals)))
        return out

    res = {"n_sessions": len(sessions), "W_bins": W,
           "commit_aligned_r2input": mean_curve(commit_curves, "r2_input"),
           "commit_aligned_auto": mean_curve(commit_curves, "auto_unique"),
           "commit_aligned_dom": mean_curve(commit_curves, "dom"),
           "stim_aligned_r2input": mean_curve(stim_curves, "r2_input"),
           "stim_aligned_auto": mean_curve(stim_curves, "auto_unique"),
           "LOCKING": {
               "swing_commit_med": round(float(np.median(swing_c)), 4), "swing_stim_med": round(float(np.median(swing_s)), 4),
               "swing_commit>stim_n": int(np.sum(np.array(swing_c[:min(len(swing_c), len(swing_s))]) > np.array(swing_s[:min(len(swing_c), len(swing_s))]))),
               "swing_paired_wilcoxon_p": round(float(stats.wilcoxon(swing_c[:min(len(swing_c), len(swing_s))], swing_s[:min(len(swing_c), len(swing_s))]).pvalue), 5) if min(len(swing_c), len(swing_s)) >= 6 else None,
               "drop_commit_med": round(float(np.median(drop_c)), 4), "drop_commit_p": round(float(stats.wilcoxon(drop_c, alternative="greater").pvalue), 5) if len(drop_c) >= 6 else None,
               "drop_stim_med": round(float(np.median(drop_s)), 4), "drop_stim_p": round(float(stats.wilcoxon(drop_s, alternative="greater").pvalue), 5) if len(drop_s) >= 6 else None,
               "n_swing": min(len(swing_c), len(swing_s)), "n_drop_c": len(drop_c)}}
    json.dump(res, open(OUT, "w"), indent=1, default=str)
    print("\n=== COMMITMENT-ALIGNED r2_input (center s rel commitment) ===", flush=True)
    for r in res["commit_aligned_r2input"]:
        print(f"  t={r['center']:+.3f}s: r2_input={r['mean']:.4f}±{r['sem']:.4f} (n_sess={r['n_sess']})", flush=True)
    print("=== STIMULUS-ALIGNED r2_input (center s rel stimulus) ===", flush=True)
    for r in res["stim_aligned_r2input"]:
        print(f"  t={r['center']:+.3f}s: r2_input={r['mean']:.4f}±{r['sem']:.4f} (n_sess={r['n_sess']})", flush=True)
    print("=== LOCKING ===", flush=True); print(json.dumps(res["LOCKING"], indent=1), flush=True)
    print("wrote", OUT, flush=True)

if __name__ == "__main__":
    main()
