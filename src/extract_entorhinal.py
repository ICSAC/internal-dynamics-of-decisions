#!/usr/bin/env python3
"""Reduce the primate entorhinal cortex mental-navigation recordings (DANDI:000897, Neupane et
al. 2024) to per-session population principal-component trajectories. For each NWB session, the
good/MUA single units are binned over a 1.0 s window from movement onset (joy1onttl) into
20 x 50 ms bins, z-scored across the selected trials, and projected onto the top R principal
components. Two conditions are extracted from the same simultaneously recorded units: inference
trials (MNAV, trial_type 3, where the map is occluded so the traversal must be mentally
constructed) and visible-map control trials (NTS, trial_type 1, or trial_type 2 when type-1
trials are sparse).

Input: NWB session files found under DATA_DIR (environment variable, default ./data).
Output: a pickle (path from argv[1], default DATA_DIR/entorhinal_trajectories.pkl) mapping each
session file name to {pctraj (trials x R x bins), trial type, step count, success, subject id,
n_units, nbins}.
"""
import os, sys, glob, pickle, warnings
warnings.filterwarnings("ignore")
import numpy as np, h5py

DATA = os.environ.get("DATA_DIR", "./data")
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(DATA, "entorhinal_trajectories.pkl")
R = 8; NB = 20; W = 1.0; DT = W/NB; MIN_UNITS = 8; MIN_TRIALS = 8
DUR_LO, DUR_HI = 0.1, 6.0; MIN_CTRL = 25

def unit_spikes(f, qset=("good", "mua")):
    u = f["processing/ecephys/units"]; st = u["spike_times"][:]; idx = u["spike_times_index"][:]
    labelcol = None
    if "quality" in u:
        ql = u["quality"][:].astype(str)
        if np.any((ql == "good") | (ql == "mua")):
            labelcol = ql
    if labelcol is None and "KSLabel" in u:
        labelcol = u["KSLabel"][:].astype(str)
    if labelcol is None:
        return []
    starts = np.concatenate([[0], idx[:-1]])
    keep = np.array([q in qset for q in labelcol])
    return [st[starts[i]:idx[i]] for i in range(len(idx)) if keep[i]]

def binned_tensor(spikes, t0, nb, dt):
    """trials x units x bins spike counts in [t0, t0+nb*dt) via vectorized searchsorted (fast)."""
    ntr = len(t0); nu = len(spikes)
    T = np.zeros((ntr, nu, nb), float)
    offs = np.arange(nb+1)*dt                       # nb+1 edges relative to t0
    E = t0[:, None] + offs[None, :]                 # ntr x (nb+1) absolute edges
    for ui, s in enumerate(spikes):
        s = np.sort(np.asarray(s))
        cs = np.searchsorted(s, E)                  # ntr x (nb+1)
        T[:, ui, :] = np.diff(cs, axis=1)
    return T

def main():
    files = sorted(glob.glob(os.path.join(DATA, "*.nwb")))
    print(f"{len(files)} sessions", flush=True); out = {}
    for fi, fn in enumerate(files):
        try:
            with h5py.File(fn, "r") as f:
                subj = f["general/subject/subject_id"][()]
                subj = subj.decode() if isinstance(subj, bytes) else str(subj)
                spikes = unit_spikes(f)
                if len(spikes) < MIN_UNITS:
                    print("  skip few units", os.path.basename(fn)); continue
                tr = f["intervals/trials"]
                tt = tr["trial_type"][:]; jon = tr["joy1onttl"][:]; joff = tr["joy1offttl"][:]
                go = tr["gocuettl"][:]; ta = tr["ta"][:]; su = tr["succ"][:]
            dur = joff - jon
            finite = np.isfinite(jon) & np.isfinite(joff) & np.isfinite(ta) & (dur > DUR_LO) & (dur < DUR_HI)
            steps = np.abs(np.round(ta)).astype(int)
            ctt = 1 if (finite & (tt == 1)).sum() >= MIN_CTRL else 2
            nts = finite & (tt == ctt); mnav = finite & (tt == 3)
            if nts.sum() < MIN_TRIALS or mnav.sum() < MIN_TRIALS:
                print("  skip few trials", os.path.basename(fn), int(nts.sum()), int(mnav.sum())); continue
            sel = np.where(finite & ((tt == ctt) | (tt == 3)))[0]
            T = binned_tensor(spikes, jon[sel], NB, DT)        # trials x units x bins
            tts = tt[sel]; sts = steps[sel]; sus = su[sel]
            # z-score units across all selected trials; drop dead units; PCA -> R
            flat = T.transpose(1, 0, 2).reshape(T.shape[1], -1)
            mu = np.nanmean(flat, 1); sd = np.nanstd(flat, 1)
            keep = (sd > 1e-9) & np.isfinite(sd) & np.isfinite(mu)
            if keep.sum() < MIN_UNITS:
                print("  skip few active", os.path.basename(fn)); continue
            Tz = (T - mu.reshape(1, -1, 1)) / sd.reshape(1, -1, 1)
            Tz = np.nan_to_num(Tz[:, keep, :])
            U_ = Tz.transpose(1, 0, 2).reshape(int(keep.sum()), -1)
            C = np.cov(U_); ev, evec = np.linalg.eigh(C); basis = evec[:, ::-1][:, :R]
            pctraj = np.einsum("ur,tub->trb", basis, Tz).astype(np.float32)   # trials x R x bins
            out[os.path.basename(fn)] = dict(pctraj=pctraj, tt=tts.astype(np.int8),
                                             steps=sts.astype(np.int16), succ=sus.astype(np.int8),
                                             ctt=int(ctt), subj=subj, n_units=int(keep.sum()), nbins=NB)
            print(f"  {fi+1}/{len(files)} {os.path.basename(fn)} subj={subj} units={int(keep.sum())} "
                  f"MNAV={int((tts==3).sum())} NTS={int((tts==ctt).sum())}", flush=True)
        except Exception as e:
            print("  ERR", os.path.basename(fn), repr(e)[:120], flush=True)
    os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
    pickle.dump(out, open(OUT, "wb"))
    print(f"saved {len(out)} sessions -> {OUT} ({os.path.getsize(OUT)/1e6:.1f} MB)", flush=True)

if __name__ == "__main__":
    main()
