#!/usr/bin/env python3
"""Poisson-clicks evidence-accumulation task (primary) + delayed-response variant (secondary).

Frozen per PREREG-CMC.md sec.1. Identical I/O for every architecture:
    input  u_t = (n_left_t, n_right_t)  per-bin Poisson click counts  (c=2)
    output      choice = sign(gamma)    single binary label per trial (generative correct side)

Constants are frozen and must not be tuned to any dynamical result.
"""
import numpy as np

# ---- frozen task constants (PREREG sec.1) ----
# Calibrated (pre-freeze, no dynamical measure touched) so Bayes-optimal commitment lands mid-trial
# and ideal final-bin accuracy ~0.95 (headroom for the >=0.90 matched-accuracy inclusion rule).
T = 40
DT = 1
COH = np.array([0.04, 0.07, 0.11, 0.18])     # coherence set
LAMBDA0 = 16.0                                # total base click rate per bin
S = 0.5                                       # coherence -> rate-fraction scale
EVIDENCE_BINS = 30                            # delayed-response: clicks in bins 1..30, zero after
DATA_SEED = 20260630
# ideal-observer accumulator (PREREG sec.5 robustness)
IO_G0 = 0.10                                  # assumed coherence for the LLR
IO_BOUND = 3.0                                # nats


def _rates(gamma_mag):
    """Poisson means for right/left given coherence magnitude."""
    lam_r = LAMBDA0 * (0.5 + S * gamma_mag)
    lam_l = LAMBDA0 * (0.5 - S * gamma_mag)
    return lam_r, lam_l


def gen_trials(n, task="primary", seed=DATA_SEED):
    """Return U (n,T,2) float click counts, y (n,) in {0,1} (1=right), gamma (n,)."""
    rng = np.random.default_rng(seed)
    # balanced coherence x side
    gmag = rng.choice(COH, size=n)
    side = rng.choice([-1.0, 1.0], size=n)          # -1=left, +1=right
    gamma = side * gmag
    lam_r, lam_l = _rates(gmag)                      # (n,)
    U = np.zeros((n, T, 2), dtype=np.float64)
    nbins = T if task == "primary" else EVIDENCE_BINS
    for t in range(nbins):
        # right-side clicks favour the side; swap means by side
        mr = np.where(side > 0, lam_r, lam_l)
        ml = np.where(side > 0, lam_l, lam_r)
        U[:, t, 1] = rng.poisson(mr)                 # n_right
        U[:, t, 0] = rng.poisson(ml)                 # n_left
    # bins >= nbins stay zero (delayed-response delay epoch)
    y = (side > 0).astype(np.int64)                  # 1 = right correct
    return U, y, gamma


def standardize(U, mu=None, sd=None):
    """Per-channel standardization using train stats (frozen at fit time)."""
    if mu is None:
        mu = U.reshape(-1, U.shape[-1]).mean(0)
        sd = U.reshape(-1, U.shape[-1]).std(0)
        sd[sd < 1e-9] = 1.0
    return (U - mu) / sd, mu, sd


def ideal_observer_commit(U_raw, task="primary"):
    """Task-defined accumulator crossing for robustness alignment (PREREG sec.5).
    Per-bin LLR under assumed coherence IO_G0; cumulative crossing of +/-IO_BOUND nats.
    Returns t_c^IO per trial (1-indexed bin), or -1 if never crosses. Uses RAW counts.
    """
    lam_r, lam_l = _rates(IO_G0)
    w = np.log(lam_r / lam_l)                         # per-click LLR weight
    # per-bin LLR contribution = (n_right - n_left) * w
    contrib = (U_raw[:, :, 1] - U_raw[:, :, 0]) * w   # (n,T)
    cum = np.cumsum(contrib, axis=1)                  # (n,T)
    crossed = np.abs(cum) >= IO_BOUND
    tc = np.full(U_raw.shape[0], -1, dtype=np.int64)
    for i in range(U_raw.shape[0]):
        idx = np.nonzero(crossed[i])[0]
        if idx.size:
            tc[i] = int(idx[0]) + 1                   # 1-indexed bin of first crossing
    return tc


if __name__ == "__main__":
    # sanity: ideal-observer final-bin accuracy on each task (NOT a network result)
    for task in ["primary", "delayed"]:
        U, y, g = gen_trials(4000, task, DATA_SEED + 1)
        # ideal final-bin: sign of total click difference
        nbins = T if task == "primary" else EVIDENCE_BINS
        diff = (U[:, :nbins, 1] - U[:, :nbins, 0]).sum(1)
        acc = (np.sign(diff) == np.where(y == 1, 1, -1)).mean()
        tc = ideal_observer_commit(U, task)
        frac_cross = (tc > 0).mean()
        in_win = ((tc >= 9) & (tc <= 32)).mean()
        print(f"{task}: ideal final-bin acc={acc:.4f}  IO-cross frac={frac_cross:.3f}  "
              f"IO in [9,32]={in_win:.3f}")
