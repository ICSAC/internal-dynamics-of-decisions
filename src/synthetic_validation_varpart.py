#!/usr/bin/env python3
"""Synthetic validation of the variance-partition autonomy read-out (autonomy_index).

On linear state-space systems with a known autonomous gain g and known input
coupling strength s, this script checks the behaviour of the robust,
cross-dataset variance-partition read-out under two input regimes: INDEP
(exogenous input) and CORR (input is a function of state).

Wants:
  A1 INDEP: autonomy_index is HIGH and rises with g (more recurrence); falls with
     input strength s.
  A2 CORR: autonomy_index is LOW (the input explains the next state) and roughly
     independent of true g, because recurrence is masked by input that carries
     state information.
  A3 Inference-vs-control test (matched input, g_inf > g_ctl): d(auto_unique)>0 and
     d(autonomy_index)>0, with a flat trial-shuffle null (small p).
  A4 NEG control (equal g, different input): d(auto_unique) ~ 0, large shuffle p.

Output: a JSON summary written to argv[1] (default ./results/validate3.json).
"""
import os, sys, json, warnings
warnings.filterwarnings("ignore")
import numpy as np
from demix import autonomy_index

SEED = 20260630; F = 6; C = 2; T = 40; NTR = 60
OUT = sys.argv[1] if len(sys.argv) > 1 else "./results/validate3.json"

def make_A(g, theta=0.35, base=0.3):
    A = np.eye(F)*base
    A[0, 0] = g*np.cos(theta); A[0, 1] = -g*np.sin(theta)
    A[1, 0] = g*np.sin(theta); A[1, 1] = g*np.cos(theta)
    return A

def gen(A, s, ntr, rng, corr=False, noise=0.01, Bfix=None, Kfix=None):
    # bounded closed loop: small feedback K + state clip, so corr trajectories stay finite.
    # Bfix/Kfix let the two conditions share the input coupling (matched input).
    B = Bfix if Bfix is not None else s*rng.standard_normal((F, C))
    K = Kfix if Kfix is not None else (0.15*rng.standard_normal((C, F)) if corr else None)
    Xs, Us = [], []
    for _ in range(ntr):
        x = rng.standard_normal(F); xs = [x]; us = []
        for t in range(T-1):
            u = (K@x+0.5*rng.standard_normal(C)) if corr else rng.standard_normal(C)
            x = np.clip(A@x+B@u+noise*rng.standard_normal(F), -50, 50); xs.append(x); us.append(u)
        us.append(np.zeros(C)); Xs.append(np.array(xs)); Us.append(np.array(us))
    return np.array(Xs), np.array(Us)

def main():
    res = {}
    # A1 / A2 grids
    for regime in ("indep", "corr"):
        rows = []
        for g in (0.60, 0.80, 0.95):
            for s in (0.0, 0.5, 1.0, 2.0):
                X, U = gen(make_A(g), s, NTR, np.random.default_rng(SEED+abs(hash((regime, g, s))) % 99999),
                           corr=(regime == "corr"))
                ai = autonomy_index(X, U)
                rows.append(dict(g=g, s=s, autonomy_index=round(ai["autonomy_index"], 3),
                                 auto_unique=round(ai["auto_unique"], 3), input_unique=round(ai["input_unique"], 3),
                                 r2_full=round(ai["r2_full"], 3), r2_input=round(ai["r2_input"], 3)))
        res[regime] = rows

    # A3 / A4 with shuffle null on auto_unique
    def cond_test(g_inf, g_ctl, s_inf, s_ctl, regime, nshuf=300):
        rng2 = np.random.default_rng(SEED+7)
        corr = (regime == "corr")
        # shared input coupling B,K across conditions (matched input)
        Bsh = max(s_inf, s_ctl) * rng2.standard_normal((F, C))
        Ksh = 0.15*rng2.standard_normal((C, F)) if corr else None
        Bi = Bsh*(s_inf/max(s_inf, s_ctl)); Bc = Bsh*(s_ctl/max(s_inf, s_ctl))
        Xi, Ui = gen(make_A(g_inf), s_inf, NTR, rng2, corr=corr, Bfix=Bi, Kfix=Ksh)
        Xc, Uc = gen(make_A(g_ctl), s_ctl, NTR, rng2, corr=corr, Bfix=Bc, Kfix=Ksh)
        ai = autonomy_index(Xi, Ui); ac = autonomy_index(Xc, Uc)
        real = ai["auto_unique"] - ac["auto_unique"]
        Xp = np.concatenate([Xi, Xc]); Up = np.concatenate([Ui, Uc]); nI = len(Xi); nulls = []
        for _ in range(nshuf):
            perm = rng2.permutation(len(Xp)); a = perm[:nI]; b = perm[nI:]
            da = autonomy_index(Xp[a], Up[a])["auto_unique"]; db = autonomy_index(Xp[b], Up[b])["auto_unique"]
            nulls.append(da-db)
        nulls = np.array(nulls); p = float((np.sum(np.abs(nulls) >= abs(real))+1)/(len(nulls)+1))
        return dict(auto_unique_inf=round(ai["auto_unique"], 3), auto_unique_ctl=round(ac["auto_unique"], 3),
                    d_auto_unique=round(real, 3), idx_inf=round(ai["autonomy_index"], 3),
                    idx_ctl=round(ac["autonomy_index"], 3), d_idx=round(ai["autonomy_index"]-ac["autonomy_index"], 3),
                    shuffle_p=round(p, 4))
    res["A3_inf_higher_g_matched_input"] = {"indep": cond_test(0.92, 0.86, 1.0, 1.0, "indep"),
                                            "corr": cond_test(0.92, 0.86, 1.0, 1.0, "corr")}
    res["A4_neg_equal_g_diff_input"] = {"indep": cond_test(0.90, 0.90, 2.0, 0.5, "indep"),
                                        "corr": cond_test(0.90, 0.90, 2.0, 0.5, "corr")}

    os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=1, default=str)
    for regime in ("indep", "corr"):
        print(f"=== {regime} (autonomy_index, auto_unique, r2_full, r2_input) ===")
        for r in res[regime]:
            print(f"  g={r['g']} s={r['s']}: idx={r['autonomy_index']} auto_uniq={r['auto_unique']} "
                  f"r2_full={r['r2_full']} r2_input={r['r2_input']}")
    print("=== A3 inf>ctl matched input (want d_auto_unique>0 indep, shuffle_p small) ===")
    print(json.dumps(res["A3_inf_higher_g_matched_input"], indent=1))
    print("=== A4 NEG control (want d_auto_unique~0, shuffle_p large) ===")
    print(json.dumps(res["A4_neg_equal_g_diff_input"], indent=1))
    print("wrote", OUT)

if __name__ == "__main__":
    main()
