#!/usr/bin/env python3
"""Frozen demix analysis per network (PREREG-CMC.md sec.5-9). RUN ONLY AFTER FREEZE.txt EXISTS.

Uses the paper's instrument demix.py UNCHANGED (sdmdc_dom rank=8 nd=2; autonomy_index defaults) plus
flow_readouts copied verbatim from analyze_commitment.py. Computes per network, own-commitment aligned
(primary) and ideal-observer aligned (robustness):
  Delta sdmdc_dom, post auto_unique/r2_input, Delta flow_balance, post settling speed, rollout MSE,
  triviality displacement/variance.
Outputs results/analysis_<task>.json (one row per network). No statistics, no verdict here.
"""
import os, sys, json, glob
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
# the paper's instrument, unchanged:
sys.path.insert(0, os.path.expanduser("~/cross-substrate-dynamics/src"))
from demix import sdmdc_dom, autonomy_index

R = 8
RANK = 8
NDELAYS = 2
W = 8
THETA = 0.40
MIN_TR = 20
ART = os.environ.get("CMC_OUT", "./artifacts")
RES = os.environ.get("CMC_RES", "./results")

# ---- flow_balance: copied verbatim from analyze_commitment.py::flow_readouts ----
from DSA import SubspaceDMDc
def to_np(M):
    return np.asarray(M.cpu() if hasattr(M, "cpu") else M)
def flow_readouts(X, U):
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


def own_commit(P):
    """first bin t (0-idx) with |p_t-0.5|>=THETA and sign holds to T; else -1."""
    d = P - 0.5
    sign = np.sign(d)
    tc = np.full(P.shape[0], -1, dtype=np.int64)
    for i in range(P.shape[0]):
        for t in range(P.shape[1]):
            if abs(d[i, t]) >= THETA and np.all(sign[i, t:] == sign[i, t]) and sign[i, t] != 0:
                tc[i] = t; break
    return tc


def windows(X, tc):
    """given trajectories X (N,T,R) and per-trial commit bin tc (0-idx), stack pre/post 8-bin windows.
    valid trial: 8 <= tc <= T-8."""
    N, T, _ = X.shape
    pre, post = [], []
    for i in range(N):
        t = tc[i]
        if t < W or t > T - W:
            continue
        pre.append(X[i, t - W:t, :])
        post.append(X[i, t:t + W, :])
    return np.array(pre), np.array(post)


def win_U(U, tc):
    N, T, _ = U.shape
    pre, post = [], []
    for i in range(N):
        t = tc[i]
        if t < W or t > T - W:
            continue
        pre.append(U[i, t - W:t, :]); post.append(U[i, t:t + W, :])
    return np.array(pre), np.array(post)


def settling_speed(Xw):
    """mean per-bin ||x_{t+1}-x_t|| over window stack (n,W,R)."""
    if len(Xw) == 0:
        return np.nan
    d = np.linalg.norm(np.diff(Xw, axis=1), axis=2)   # (n,W-1)
    return float(d.mean())


def state_var(Xw):
    if len(Xw) == 0:
        return np.nan
    flat = Xw.reshape(-1, Xw.shape[-1])
    return float(np.trace(np.cov(flat.T)))


def rollout_mse(Xpre, Xpost, nd=NDELAYS, ridge=1e-2):
    """fit delay-embedded autonomous A on pre (LSQ), free-run on post, normalized MSE (PREREG sec.8.3)."""
    if len(Xpre) == 0 or len(Xpost) == 0:
        return np.nan
    P1, P2 = [], []
    for Z in Xpre:
        H = np.concatenate([Z[nd - 1 - k: len(Z) - k] for k in range(nd)], axis=1)
        P1.append(H[:-1]); P2.append(H[1:])
    X1 = np.concatenate(P1); X2 = np.concatenate(P2); d = X1.shape[1]
    try:
        A = np.linalg.solve(X1.T @ X1 + ridge * np.eye(d), X1.T @ X2).T
    except np.linalg.LinAlgError:
        return np.nan
    errs = []
    for Z in Xpost:
        cur = np.concatenate([Z[nd - 1 - k] for k in range(nd)])   # [x_{nd-1},x_{nd-2},...]
        true = Z[nd:]                                              # x from step nd on
        pred = []
        for _ in range(len(true)):
            cur = A @ cur
            pred.append(cur[:Z.shape[1]])
        pred = np.array(pred)
        if len(pred) == 0:
            continue
        var = np.var(true) + 1e-9
        errs.append(np.mean((pred - true) ** 2) / var)
    return float(np.mean(errs)) if errs else np.nan


def analyze_net(X, P, U, tc_io, meta):
    out = dict(meta)
    for tag, tc in [("own", own_commit(P)), ("io", tc_io)]:
        Xpre, Xpost = windows(X, tc)
        Upre, Upost = win_U(U, tc)
        n_pre, n_post = len(Xpre), len(Xpost)
        rec = dict(n_pre=n_pre, n_post=n_post)
        if n_pre >= MIN_TR and n_post >= MIN_TR:
            sp = sdmdc_dom(Xpre, Upre, rank=RANK, nd=NDELAYS)
            so = sdmdc_dom(Xpost, Upost, rank=RANK, nd=NDELAYS)
            rec["sdmdc_pre"] = sp; rec["sdmdc_post"] = so
            rec["dsdmdc"] = (so - sp) if (np.isfinite(sp) and np.isfinite(so)) else np.nan
            ai = autonomy_index(Xpost, Upost)
            rec["auto_unique_post"] = ai["auto_unique"]; rec["r2_input_post"] = ai["r2_input"]
            fp = flow_readouts(Xpre, Upre); fo = flow_readouts(Xpost, Upost)
            fb_pre = fp[1] if fp else np.nan; fb_post = fo[1] if fo else np.nan
            rec["flow_pre"] = fb_pre; rec["flow_post"] = fb_post
            rec["dflow"] = (fb_post - fb_pre) if (np.isfinite(fb_pre) and np.isfinite(fb_post)) else np.nan
            rec["speed_post"] = settling_speed(Xpost)
            rec["speed_pre"] = settling_speed(Xpre)
            rec["var_post"] = state_var(Xpost)
            rec["rollout_mse"] = rollout_mse(Xpre, Xpost)
            rec["disp_post"] = rec["speed_post"]; rec["disp_pre"] = rec["speed_pre"]
        out[tag] = rec
    return out


def main():
    task = sys.argv[1] if len(sys.argv) > 1 else "primary"
    os.makedirs(RES, exist_ok=True)
    shared = np.load(os.path.join(ART, f"shared_{task}.npz"))
    U = shared["U"]; tc_io = shared["tc_io"] - 1  # io stored 1-indexed; -1 invalid stays <0 -> excluded
    tc_io = np.where(shared["tc_io"] > 0, shared["tc_io"] - 1, -1)
    table = json.load(open(os.path.join(ART, f"acc_table_{task}.json")))
    rows = []
    for m in table:
        f = os.path.join(ART, f"net_{task}_{m['arch']}_{m['seed']}.npz")
        d = np.load(f)
        X = d["X"]; P = d["P"]
        meta = {k: m[k] for k in ("arch", "family", "seed", "task", "width", "nparams", "test_acc")}
        r = analyze_net(X, P, U, tc_io, meta)
        rows.append(r)
        own = r["own"]
        print(f"{task} {m['arch']:4s} s{m['seed']} acc={m['test_acc']:.3f} "
              f"n_post={own.get('n_post',0):4d} dsdmdc={own.get('dsdmdc',float('nan')):+.4f} "
              f"auto_u={own.get('auto_unique_post',float('nan')):+.4f} "
              f"disp_post={own.get('disp_post',float('nan')):.4f}", flush=True)
    json.dump(rows, open(os.path.join(RES, f"analysis_{task}.json"), "w"), indent=1, default=str)
    print("wrote", os.path.join(RES, f"analysis_{task}.json"), flush=True)


if __name__ == "__main__":
    main()
