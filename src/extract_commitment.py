#!/usr/bin/env python3
"""Align the perceptual-decision Neuropixels recordings to per-trial commitment time
(Luo et al. 2025; Dryad 10.5061/dryad.sj3tx96dm and the accompanying decision_dynamics_commitment
release). For each session, the simultaneously recorded units are pooled and joined to the
per-trial commitment-time table (nTc.csv). The script builds commitment-aligned and
stimulus(stereoclick)-aligned population principal-component trajectories by absolute-time binning
(25 ms bins; commitment window [-0.35, 0.35] s, stimulus window [0, 0.70] s) together with the
accumulated-evidence input derived from the left/right click times. Per-session unit activity is
z-scored before projection onto the top R principal components, and the evidence channels are
z-scored per session.

Input: MATLAB session files (*.mat) under DATA_DIR/recordings and the commitment-time table
DATA_DIR/nTc.csv (DATA_DIR from the environment, default ./data). Output: a pickle (path from
argv[1], default DATA_DIR/commitment_aligned.pkl) mapping each session to its commitment- and
stimulus-aligned PC trajectories, the aligned evidence inputs, movement time relative to
commitment, unit counts, and bin offsets. Optional argv[2] sets the worker-pool size.
"""
import os, sys, glob, pickle, warnings, csv
warnings.filterwarnings("ignore")
import numpy as np
import scipy.io as sio
from multiprocessing import Pool

DATA = os.environ.get("DATA_DIR", "./data")
REC = os.path.join(DATA, "recordings")
NTC = os.path.join(DATA, "nTc.csv")
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(DATA, "commitment_aligned.pkl")
DT = 0.025; R = 8
C_LO, C_HI = -0.35, 0.35          # commitment-aligned window (s)
S_LO, S_HI = 0.0, 0.70            # stimulus-aligned window (s)
NPROC = int(sys.argv[2]) if len(sys.argv) > 2 else (os.cpu_count() or 4)

def unwrap(x):
    a = np.asarray(x)
    while a.dtype == object and a.size == 1:
        a = np.asarray(a.flat[0])
    return a

def flat_str(x):
    return [str(s) for s in unwrap(x).ravel()]

def is_striatum(area):
    a = area.lower()
    return any(k in a for k in ("str", "nac", "caud", "putamen", "ads", "dms", "dls", " ts", "tail"))

def load_ntc():
    rows = {}
    for r in csv.DictReader(open(NTC)):
        rows.setdefault(r["recording_id"], []).append(r)
    return rows

def bin_counts(spike_times_sorted, edges):
    cs = np.searchsorted(spike_times_sorted, edges)
    return np.diff(cs)

def evidence_series(lb_abs, rb_abs, centers, lo_edges, hi_edges):
    # cumulative net + increment at bins
    cum = np.array([np.sum(rb_abs < t) - np.sum(lb_abs < t) for t in centers], float)
    incr = np.array([(np.sum((rb_abs >= a) & (rb_abs < b)) - np.sum((lb_abs >= a) & (lb_abs < b)))
                     for a, b in zip(lo_edges, hi_edges)], float)
    return np.stack([cum, incr], axis=1)

def work(item):
    fn, ntc_rows = item
    try:
        d = sio.loadmat(fn, squeeze_me=True, struct_as_record=False)
        C, T = d["Cells"], d["Trials"]
        sp = np.atleast_1d(C.raw_spike_time_s)        # per-unit spike-time arrays
        units = [np.sort(np.asarray(sp[i]).ravel().astype(float)) for i in range(len(sp))]
        nU = len(units)
        if nU < 12:
            return None
        striat = np.zeros(nU, bool)                   # area assignment deferred: unit area labels are MATLAB MCOS objects not readable via scipy.io
        lb_all = np.atleast_1d(T.leftBups); rb_all = np.atleast_1d(T.rightBups)
        nbc = int(round((C_HI - C_LO) / DT)); nbs = int(round((S_HI - S_LO) / DT))
        commit_off = C_LO + DT * np.arange(nbc)      # bin starts rel commitment
        stim_off = S_LO + DT * np.arange(nbs)
        Cc, Sc, Ec, Es, mv = [], [], [], [], []      # per-trial tensors
        for r in ntc_rows:
            if r["commitment_time_s"] in ("NaN", ""):
                continue
            ci = int(r["index_in_Trials"]) - 1
            if ci < 0 or ci >= lb_all.shape[0]:
                continue
            s = float(r["stereoclick_time_s"]); c = float(r["commitment_time_s"])
            mvt = float(r["spoke_s"]) if r["spoke_s"] not in ("NaN", "") else np.nan
            lb = np.atleast_1d(np.asarray(lb_all[ci]).astype(float)).ravel() + s   # absolute
            rb = np.atleast_1d(np.asarray(rb_all[ci]).astype(float)).ravel() + s
            # commitment-aligned counts (units x nbc)
            ce_lo = c + commit_off; ce = np.append(ce_lo, c + commit_off[-1] + DT)
            se_lo = s + stim_off; se = np.append(se_lo, s + stim_off[-1] + DT)
            Mc = np.stack([bin_counts(u, ce) for u in units], axis=0)   # units x nbc
            Ms = np.stack([bin_counts(u, se) for u in units], axis=0)
            Cc.append(Mc.T); Sc.append(Ms.T)                            # (nbc,units)
            Ec.append(evidence_series(lb, rb, ce_lo + DT/2, ce_lo, ce_lo + DT))
            Es.append(evidence_series(lb, rb, se_lo + DT/2, se_lo, se_lo + DT))
            mv.append(mvt - c)                                          # movement time rel commitment
        if len(Cc) < 20:
            return None
        Cc = np.stack(Cc, 0); Sc = np.stack(Sc, 0)        # (ntr, nb, units)
        Ec = np.stack(Ec, 0); Es = np.stack(Es, 0)
        # z-score units on pooled (commit+stim) bins, PCA R
        pooled = np.concatenate([Cc.reshape(-1, nU), Sc.reshape(-1, nU)], 0)
        mu = pooled.mean(0); sd = pooled.std(0); sd[sd < 1e-9] = 1.0
        keep = sd > 1e-9
        Czs = (Cc - mu) / sd; Szs = (Sc - mu) / sd
        flat = np.concatenate([Czs.reshape(-1, nU), Szs.reshape(-1, nU)], 0)
        U, S, Vt = np.linalg.svd(flat - flat.mean(0), full_matrices=False)
        basis = Vt[:R].T
        commit_traj = (Czs @ basis).astype(np.float32)    # (ntr, nbc, R)
        stim_traj = (Szs @ basis).astype(np.float32)
        # z-score evidence channels (per session)
        for E in (Ec, Es):
            f = E.reshape(-1, 2); m = f.mean(0); s2 = f.std(0); s2[s2 < 1e-9] = 1.0
            E[:] = (E - m) / s2
        return (os.path.basename(fn).replace(".mat", ""),
                dict(commit_traj=commit_traj, stim_traj=stim_traj,
                     ev_commit=Ec.astype(np.float32), ev_stim=Es.astype(np.float32),
                     mv_rel_commit=np.array(mv, np.float32), n_units=int(nU),
                     n_frontal=int((~striat).sum()), n_striatum=int(striat.sum()),
                     areas={}, n_trials=int(len(Cc)),
                     commit_off=commit_off.astype(np.float32), stim_off=stim_off.astype(np.float32), dt=DT))
    except Exception as e:
        return (os.path.basename(fn), {"error": repr(e)[:160]})

def main():
    ntc = load_ntc()
    files = sorted(glob.glob(os.path.join(REC, "*.mat")))
    items = [(f, ntc.get(os.path.basename(f).replace(".mat", ""), [])) for f in files]
    items = [(f, r) for f, r in items if r]
    print(f"{len(items)} sessions with commitment times; Pool({NPROC})", flush=True)
    out = {}; errs = 0
    with Pool(NPROC) as pl:
        for res in pl.imap_unordered(work, items):
            if res is None:
                continue
            k, v = res
            if "error" in v:
                errs += 1; print("  ERR", k, v["error"][:80], flush=True); continue
            out[k] = v
            if len(out) % 20 == 0:
                print(f"  {len(out)} done", flush=True)
    os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
    pickle.dump(out, open(OUT, "wb"))
    tot_tr = sum(v["n_trials"] for v in out.values())
    tot_fr = sum(v["n_frontal"] for v in out.values()); tot_st = sum(v["n_striatum"] for v in out.values())
    print(f"saved {len(out)} sessions ({errs} errs) -> {OUT}; total committed trials={tot_tr}; "
          f"units frontal={tot_fr} striatum={tot_st}", flush=True)
    # area inventory
    allareas = __import__("collections").Counter()
    for v in out.values():
        allareas.update(v["areas"])
    print("AREAS:", dict(allareas.most_common(20)), flush=True)

if __name__ == "__main__":
    main()
