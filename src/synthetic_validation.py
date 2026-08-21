#!/usr/bin/env python3
"""Synthetic validation of the autonomous-versus-input demix using the canonical
SubspaceDMDc (N4SID) estimator.

On linear state-space systems with a known autonomous gain g and a known input
coupling of strength s, this script checks that the autonomous-eigenvalue
read-out recovers g and is invariant to s when the input is exogenous, while a
plain (non-subspace) DMD eigenvalue is not. It runs both an exogenous-input
regime (INDEP) and a state-correlated-input regime (CORR).

Read-outs per condition:
  AUTONOMOUS recurrence = dom|lambda|(A_v)  [latent autonomous operator]
  BALANCE = mean||A_v z_t||^2 / (mean||A_v z_t||^2 + mean||B_v u_t||^2)  in [0,1]
            with z_t, u_t taken from get_training_latent_states() so both terms
            live in the demixed latent space.

Validations:
  V1 dom|lambda|(A_v) ~ g, invariant to input strength s.
  V2 plain DMD dom|lambda| drifts with s (contaminated).
  V3 balance falls with s at fixed g.
  V4 g_inf=.92 > g_ctl=.86 at matched input -> d_dom>0 & d_balance>0, trial-shuffle null flat.
  V5 NEG control: g_inf==g_ctl, s_inf>s_ctl -> d_dom~0 (input difference must not fool the demix).
  V6 delay/rank robustness grid.

Output: a JSON summary written to argv[1] (default ./results/inputdsa_synthetic2.json).
"""
import os, sys, json, warnings
warnings.filterwarnings("ignore")
import numpy as np
from DSA import DMD, SubspaceDMDc

SEED = 20260630
F = 6; C = 2; T = 40; NTR = 60
OUT = sys.argv[1] if len(sys.argv) > 1 else "./results/inputdsa_synthetic2.json"

def make_A(g, theta=0.35, base=0.3):
    A = np.eye(F) * base
    A[0, 0] = g*np.cos(theta); A[0, 1] = -g*np.sin(theta)
    A[1, 0] = g*np.sin(theta); A[1, 1] = g*np.cos(theta)
    return A

def gen(A, s, ntr, rng, corr=False, noise=0.01):
    B = s * rng.standard_normal((F, C))
    K = 0.5 * rng.standard_normal((C, F)) if corr else None
    Xs, Us = [], []
    for _ in range(ntr):
        x = rng.standard_normal(F); xs = [x]; us = []
        for t in range(T-1):
            u = (K @ x + 0.5*rng.standard_normal(C)) if corr else rng.standard_normal(C)
            x = A @ x + B @ u + noise*rng.standard_normal(F)
            xs.append(x); us.append(u)
        us.append(np.zeros(C)); Xs.append(np.array(xs)); Us.append(np.array(us))
    return np.array(Xs), np.array(Us)

def to_np(M):
    return np.asarray(M.cpu() if hasattr(M, "cpu") else M)

def dom(M):
    M = to_np(M)
    return float(np.max(np.abs(np.linalg.eigvals(M))))

def fit_sdmdc(X, U, rank=F, nd=1):
    d = SubspaceDMDc(X, control_data=U, n_delays=nd, rank=rank, device="cpu", verbose=False); d.fit()
    return d

def readouts(X, U, rank=F, nd=1):
    """returns (dom|lambda|(A_v), balance) robustly; np.nan on failure."""
    try:
        d = fit_sdmdc(X, U, rank=rank, nd=nd)
        Av = to_np(d.A_v); Bv = to_np(d.B_v)
        dl = float(np.max(np.abs(np.linalg.eigvals(Av))))
        z, znext, u = d.get_training_latent_states(return_aligned=True)
        z = to_np(z); u = to_np(u)
        # align B_v columns to u dims
        m = min(Bv.shape[1], u.shape[1])
        auto = np.mean(np.sum((z @ Av.T)**2, axis=1))
        inp = np.mean(np.sum((u[:, :m] @ Bv[:, :m].T)**2, axis=1))
        bal = float(auto/(auto+inp+1e-12))
        return dl, bal
    except Exception as e:
        return np.nan, np.nan

def plain_dmd_dom(X, nd=1):
    try:
        d = DMD(X, n_delays=nd, rank=F*nd, device="cpu", verbose=False); d.fit()
        return dom(d.A_v)
    except Exception:
        return np.nan

def main():
    res = {"params": {"F": F, "C": C, "T": T, "NTR": NTR, "seed": SEED, "estimator": "SubspaceDMDc(rank=F)"}}

    # V1/V2/V3
    res["V1_V2_V3"] = {}
    for regime in ("indep", "corr"):
        rows = []
        for g in (0.60, 0.80, 0.95):
            for s in (0.0, 0.5, 1.0, 2.0):
                A = make_A(g)
                X, U = gen(A, s, NTR, np.random.default_rng(SEED+abs(hash((regime, g, s))) % 99999),
                           corr=(regime == "corr"))
                dl, bal = readouts(X, U)
                pdom = plain_dmd_dom(X)
                rows.append(dict(g_true=g, s=s, sdmdc_dom=round(dl, 4), dmd_dom=round(pdom, 4),
                                 balance=round(bal, 4)))
        res["V1_V2_V3"][regime] = rows
        for g in (0.60, 0.80, 0.95):
            sub = [r for r in rows if r["g_true"] == g and np.isfinite(r["sdmdc_dom"])]
            if sub:
                res.setdefault("V1_sdmdc_invariance_spread", {})[f"{regime}_g{g}"] = round(
                    max(r["sdmdc_dom"] for r in sub) - min(r["sdmdc_dom"] for r in sub), 4)
                res.setdefault("V1_sdmdc_mean_vs_true", {})[f"{regime}_g{g}"] = round(
                    np.mean([r["sdmdc_dom"] for r in sub]), 4)
            subp = [r for r in rows if r["g_true"] == g and np.isfinite(r["dmd_dom"])]
            if subp:
                res.setdefault("V2_dmd_contamination_spread", {})[f"{regime}_g{g}"] = round(
                    max(r["dmd_dom"] for r in subp) - min(r["dmd_dom"] for r in subp), 4)

    # V4 / V5 with trial-shuffle null
    def cond_test(g_inf, g_ctl, s_inf, s_ctl, regime, nshuf=300):
        rng2 = np.random.default_rng(SEED+7)
        Xi, Ui = gen(make_A(g_inf), s_inf, NTR, rng2, corr=(regime == "corr"))
        Xc, Uc = gen(make_A(g_ctl), s_ctl, NTR, rng2, corr=(regime == "corr"))
        di, bi = readouts(Xi, Ui); dc, bc = readouts(Xc, Uc)
        Xp = np.concatenate([Xi, Xc], 0); Up = np.concatenate([Ui, Uc], 0); nI = len(Xi)
        real = di - dc; nulls = []
        for _ in range(nshuf):
            perm = rng2.permutation(len(Xp)); a = perm[:nI]; b = perm[nI:]
            da, _ = readouts(Xp[a], Up[a]); db, _ = readouts(Xp[b], Up[b])
            if np.isfinite(da) and np.isfinite(db):
                nulls.append(da-db)
        nulls = np.array(nulls)
        p = float((np.sum(np.abs(nulls) >= abs(real))+1)/(len(nulls)+1)) if nulls.size else None
        return dict(sdmdc_dom_inf=round(di, 4), sdmdc_dom_ctl=round(dc, 4), d_dom=round(real, 4),
                    balance_inf=round(bi, 4), balance_ctl=round(bc, 4), d_balance=round(bi-bc, 4),
                    shuffle_p=p, n_null=int(nulls.size))

    res["V4_inf_higher_autonomy_matched_input"] = {
        "indep": cond_test(0.92, 0.86, 1.0, 1.0, "indep"),
        "corr":  cond_test(0.92, 0.86, 1.0, 1.0, "corr"),
    }
    res["V5_neg_equal_autonomy_diff_input"] = {
        "indep": cond_test(0.90, 0.90, 2.0, 0.5, "indep"),
        "corr":  cond_test(0.90, 0.90, 2.0, 0.5, "corr"),
    }

    # V6 delay/rank robustness on a g=0.92 corr system
    A = make_A(0.92); X, U = gen(A, 1.0, NTR, np.random.default_rng(SEED+11), corr=True)
    grid = {}
    for nd in (1, 2, 3):
        for rank in (F, F+2):
            dl, bal = readouts(X, U, rank=rank, nd=nd)
            grid[f"nd{nd}_rank{rank}"] = dict(dom=round(dl, 4) if np.isfinite(dl) else None,
                                              balance=round(bal, 4) if np.isfinite(bal) else None)
    res["V6_delay_rank_grid_true0.92"] = grid

    os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=1, default=str)
    print("=== V1 SubspaceDMDc autonomous-dom: mean vs true g (want ~equal), spread across s (want ~0) ===")
    print("mean_vs_true:", json.dumps(res.get("V1_sdmdc_mean_vs_true", {})))
    print("invariance_spread:", json.dumps(res.get("V1_sdmdc_invariance_spread", {})))
    print("=== V2 plain-DMD contamination spread (want >> V1 spread) ===")
    print(json.dumps(res.get("V2_dmd_contamination_spread", {})))
    print("=== corr-regime table ===")
    for r in res["V1_V2_V3"]["corr"]:
        print(f"  g={r['g_true']} s={r['s']}: sdmdc_dom={r['sdmdc_dom']} dmd_dom={r['dmd_dom']} balance={r['balance']}")
    print("=== V4 inference>control, matched input (want d_dom>0, d_balance>0, shuffle_p small) ===")
    print(json.dumps(res["V4_inf_higher_autonomy_matched_input"], indent=1))
    print("=== V5 NEG control (want d_dom~0, shuffle_p large) ===")
    print(json.dumps(res["V5_neg_equal_autonomy_diff_input"], indent=1))
    print("=== V6 delay/rank grid (true 0.92) ===")
    print(json.dumps(res["V6_delay_rank_grid_true0.92"], indent=1))
    print("wrote", OUT)

if __name__ == "__main__":
    main()
