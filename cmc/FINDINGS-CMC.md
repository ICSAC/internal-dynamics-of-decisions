# FINDINGS-CMC.md — Computation-matched architecture control

**Verdict: (a) PRIMARY CONFIRMS — qualified.** The paper's own commitment-attractor measure (autonomous-
eigenvalue strengthening at commitment, `Δ sdmdc_dom`) **does diagnose recurrent-vs-feedforward architecture
when the input→output computation is held fixed**, robustly and reproducibly. The co-primary variance-balance
measure (`auto_unique`) does **not** reach significance — so the diagnostic signature is the *eigenvalue /
autonomous-flow* family of read-outs, **not** the variance partition. Reported exactly as it landed, against
the pre-registration frozen in `PREREG-CMC.md` / `FREEZE.txt` (frozen 2026-06-30T22:05:47Z, before any
dynamical measure was computed). No metric-hunting: the lead metric was the paper's instrument, fixed in
advance; the three secondaries were a frozen, Holm-corrected list reported regardless of outcome.

This is a controlled **model-organism experiment**, not a claim about real brains. It establishes only that
the paper's eigenvalue-based commitment-attractor measure is *the kind of measure that distinguishes
architecture under matched computation*.

---

## 1. Matched accuracy (the unfolding precondition) — met cleanly

All six architectures were trained to the identical I/O mapping (Poisson-clicks → binary choice) and reached
indistinguishable final-bin test accuracy. **Every network passed the ≥0.90, ±0.02 inclusion rule; 0 excluded,
0 trivially-frozen, across all four runs (2 tasks × 2 independent seed sets = 144 networks).**

| Task | acc range (seeds 0–5) | mean | acc range (seeds 6–11, replication) | params/arch |
|------|----------------------|------|--------------------------------------|-------------|
| Primary (clicks, evidence all 40 bins) | 0.9433–0.9493 | 0.9465 | 0.9425–0.9493 | 60.7k–67.8k |
| Delayed (evidence 1–30, zero-input delay 31–40) | 0.9255–0.9323 | 0.9289 | 0.9247–0.9317 | 60.7k–67.8k |

Parameter budgets are within the frozen [60k,160k] band and within 1.13× across all six architectures.
Both families (recurrent: RNN/GRU/LSTM; feedforward: MLP/TCN/causal-Transformer) match the I/O at matched
accuracy → the Doerig-et-al. unfolding precondition the main paper could not meet for brain-vs-LLM is met
here by construction.

## 2. PRIMARY result — `Δ sdmdc_dom` (autonomous-eigenvalue change at commitment), own-commitment aligned

One-sided Mann–Whitney U, recurrent > feedforward, network as unit (n=18 vs 18 per run):

| Run | p | mean Δsdmdc rec | mean Δsdmdc ff | CI(mean diff) | verdict |
|-----|----|----------------:|---------------:|---------------|---------|
| Primary, seeds 0–5 | **0.0129** | +0.0470 | +0.0129 | [+0.008, +0.061] | CONFIRM |
| Primary, seeds 6–11 (replication) | **0.0014** | — | — | (med +0.051 vs +0.010) | CONFIRM |
| Delayed, seeds 0–5 | **0.00013** | (med +0.090) | (med +0.0003) | [+0.052, +0.108] | CONFIRM |
| Delayed, seeds 6–11 (replication) | **<0.00001** | (med +0.097) | (med −0.007) | — | CONFIRM |

The lead metric separates the families in the predicted direction in **all four independent runs**, survives
the triviality guard unchanged (no frozen nets; recurrent settling is a genuine attractor, not degeneracy),
and reproduces on fresh seeds. **This is the computation-matched confirmation the main paper lacked.**

## 3. Co-primary and secondaries — which measures are diagnostic

Co-primary **`auto_unique`** (variance-partition autonomy): p = 0.16 / 0.059 / 0.14 / 0.075 across the four
runs — a **consistent same-direction trend that never reaches α=0.05.** It is **not** diagnostic here. Reason
visible in the per-arch table: the feedforward TCN scores the *highest* `auto_unique` of any architecture
(0.78 primary, 0.74 delayed), because a causal conv over autocorrelated cumulative-click input produces
state that is linearly self-predictable even without recurrence — exactly the failure mode the eigenvalue
read-out (`sdmdc_dom`, designed to recover the true autonomous gain invariant to input) is meant to avoid.

Frozen secondaries (Holm-corrected, reported regardless of outcome):

| Secondary (direction) | Primary p_holm | Delayed p_holm | separates? |
|-----------------------|---------------:|---------------:|-----------|
| Δ flow_balance (rec>ff) — paper's autonomous-flow fraction | **0.0004** | **5e-7** | YES (both) |
| settling speed (rec<ff) — post-commitment slowing | **0.003** | **0.0002** | YES (both) |
| autonomous rollout MSE (rec<ff) | 0.18 | 0.11 | no |

**Pattern:** the two measures that confirm alongside `Δ sdmdc_dom` are *also* autonomous-attractor measures
(eigenvalue-flow fraction; trajectory settling). The variance-partition (`auto_unique`) and the input-free
rollout do not. The diagnostic signature under matched computation is **autonomous-attractor strengthening /
settling at commitment** — the exact measure family the main paper leads with (`dom|λ|` of the autonomous
operator, in both the brain and the LLM legs).

## 4. Per-architecture detail (seeds 0–5, own-commitment alignment)

```
PRIMARY                                          DELAYED
arch fam   acc   Δsdmdc auto_u  Δflow disp      arch fam   acc   Δsdmdc auto_u  Δflow disp
RNN  R   0.9476 +0.0165 0.675 +0.064 1.339      RNN  R   0.9290 +0.0055 0.678 +0.104 0.923
GRU  R   0.9470 +0.0266 0.710 +0.080 0.511      GRU  R   0.9301 +0.0926 0.720 +0.137 0.378
LSTM R   0.9472 +0.0979 0.713 +0.120 0.351      LSTM R   0.9302 +0.1148 0.755 +0.138 0.236
MLP  F   0.9451 +0.0098 0.674 +0.080 1.190      MLP  F   0.9288 -0.0002 0.692 +0.074 0.923
TCN  F   0.9464 +0.0536 0.777 +0.031 0.964      TCN  F   0.9284 -0.0091 0.742 +0.038 0.855
TFM  F   0.9454 -0.0248 0.507 +0.031 4.542      TFM  F   0.9269 -0.0173 0.551 +0.037 3.603
```

## 5. Honest caveats (these qualify the confirm — do not drop them in the paper)

1. **It is a family-level distributional effect, not a clean per-instance split.** The separation is carried
   by LSTM (strongest attractor) at the top and the causal Transformer (negative Δsdmdc, large persistent
   displacement) at the bottom. In the *middle*, the feedforward TCN (Δsdmdc +0.054 primary) outscores the
   recurrent RNN and GRU. The claim that survives is "recurrence shifts the distribution of the
   attractor measure upward," **not** "every recurrent net beats every feedforward net."
2. **Vanilla RNN is weak**, especially on the delayed task (Δsdmdc +0.006), consistent with its known
   difficulty holding memory across the zero-input delay. The gated recurrent nets (GRU/LSTM) carry the
   recurrent family.
3. **Alignment specificity.** On the always-on primary task, the confirm holds for each network's *own*
   commitment alignment (p=0.013) but **not** for the task-defined ideal-observer alignment (p=0.21). On the
   delayed task it holds for **both** (own p=0.00013; ideal-observer p=0.0026). The signature is cleanest
   when the post-commitment window contains **zero input** (delayed) — i.e. when there is no exogenous drive
   to mimic autonomy in the feedforward nets. This is the expected and most interpretable regime.
4. **The variance-balance read-out (`auto_unique`) is not diagnostic** under matched computation (§3); the
   architecture claim must rest on the **eigenvalue / autonomous-flow** measures, which is what the paper
   already leads with.

## 6. Integration action (outcome (a), qualified)

- **Add a Results subsection + figure** ("Computation-matched architecture control"): under a held-fixed
  input→output computation at matched accuracy, the paper's autonomous-eigenvalue commitment-attractor
  measure (`Δ sdmdc_dom`) separates recurrent from feedforward architectures (4/4 runs, p ≤ 0.013;
  replicated on fresh seeds), corroborated by the autonomous-flow fraction and post-commitment settling.
  Figure: per-arch `Δ sdmdc_dom` with family contrast + the matched-accuracy table.
- **Discussion — retire the "not computation-matched" limitation, precisely.** The limitation is retired
  **for the eigenvalue/attractor measure the paper actually uses** (`dom|λ|` of the autonomous operator),
  with the §5 caveats stated: distributional/family-level, gated-recurrent-carried, cleanest under input-free
  post-commitment epochs. Do **not** claim the variance-balance measure is computation-diagnostic.
- **Brain↔LLM consistency.** The control validates the *same measure family* the paper leads with in both
  substrates (autonomous-eigenvalue strengthening), so it directly supports the headline cross-substrate
  contrast rather than a side measure.
- **Framing.** Keep the model-organism caveat: this shows the measure *can* diagnose architecture under
  matched computation; it is not itself evidence about brains.

## 7. Provenance / reproducibility

- Pre-registration + immutable hashes: `PREREG-CMC.md`, `FREEZE.txt` (frozen before any demix; instrument
  `demix.py` used unchanged, hash recorded).
- Code (frozen): `code/{task,models,train,analyze_cmc,stats_cmc}.py`; replication driver `code/rep_seeds2.py`
  (orchestration only — calls the frozen pipeline with the pre-registered SEEDS2=[6..11]).
- Results pulled off the ephemeral pod: `results_pod/{results,results_rep,artifacts,artifacts_rep}/*.json`
  (per-network analysis, family statistics, matched-accuracy tables, both seed sets). Raw trained
  trajectories (1.4 GB) were left on the ephemeral pod; the frozen code + fixed seeds (`DATA_SEED=20260630`)
  regenerate them deterministically.
- Compute: RTX PRO 6000 pod; ~3 s/network training; `OMP/OPENBLAS_NUM_THREADS=1` pinned for the demix loops.

## 8. One-line summary

Held the computation fixed and varied only architecture at matched accuracy; the paper's own autonomous-
eigenvalue commitment-attractor measure separated recurrent from feedforward nets in 4/4 pre-registered runs
(p ≤ 0.013, replicated), while the variance-balance measure did not — so the implementation claim is **earned
for the eigenvalue/attractor measure**, with honest family-level and alignment caveats. No metric-hunting.
