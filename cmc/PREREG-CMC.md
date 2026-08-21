# PREREG-CMC.md — Pre-registered computation-matched architecture control

**Status: FROZEN before any dynamical-measure (demix) computation on trained networks.**
This document fixes every design, alignment, metric, statistical, and decision choice in advance.
It is registered per `PREREG-computation-matched-control-PROMPT.md`. The companion code is committed
alongside and its hashes are recorded in `FREEZE.txt`. Training networks and confirming matched
accuracy is permitted before freezing (it does not touch the dynamical measures); **no `sdmdc_dom`,
`autonomy_index`, `flow_balance`, settling, or rollout value is computed, inspected, or plotted until
this file and the code are frozen.** Whatever the test returns is reported (verdict (a), (b), or (c)).

This is a controlled **model-organism experiment**: does the paper's own dynamical instrument
distinguish recurrent from feedforward architectures *when the input→output computation is held
fixed*? It is **not** a claim about real brains; it tells us only whether the signature is the *kind*
of measure that diagnoses architecture. It will be framed exactly that way in any writeup.

---

## 0. The mistake this design prevents

A prior attempt (`toymodel/ABANDONED.md`) was rigged: when variance-share did not separate the
architectures, a settling metric was tried, then a third — stopping when one separated. That is
metric-hunting. Here the **primary metric is fixed to the paper's own instrument** (`src/demix.py`,
used unchanged). Secondary metrics are a frozen, multiplicity-corrected list of exactly three, each
with a pre-stated direction and kill condition. Nothing is chosen, added, or dropped after any result
is seen. A secondary that separates while the primary does not is reported as "the paper's metric is
**not** the diagnostic one" — never spun as confirmation.

---

## 1. Task (identical I/O for every network)

**Exogenous input** at each task-time bin t: `u_t = (n_left_t, n_right_t)`, the per-bin left/right
Poisson click counts (`c = 2`). This is the input regressor used by the demix, exactly analogous to
the brain's aligned click evidence `ev`.

**Target output:** a single binary choice = `sign(sum_t (n_right_t − n_left_t))` (the side with more
total clicks over the trial). One label per trial. **The I/O mapping `{u_1..u_T} → choice` is identical
for all architectures and both tasks.**

### 1a. PRIMARY task — Poisson-clicks evidence accumulation
- `T = 40` bins, bin size `dt = 1` (abstract units). Clicks present in **all 40 bins**.
- Per trial: draw stimulus strength `gamma ~ Uniform{±g}` over a fixed coherence set
  `g ∈ {0.04, 0.07, 0.11, 0.18}` (sign = correct side), base rate `lambda0 = 16` clicks/bin total.
  (Calibrated pre-freeze on the Bayes-optimal observer alone — no network, no dynamical measure — so
  ideal final-bin accuracy ≈ 0.95 and the optimal commitment lands mid-trial; constants are now fixed.)
  Left/right Poisson means: `lambda_R = lambda0·(0.5 + s·gamma_mag)`, `lambda_L = lambda0·(0.5 − s·gamma_mag)`
  where `s = 0.5` scales coherence into a rate fraction and `gamma_mag = |gamma|`. Correct side = `sign(gamma)`.
  (These constants are chosen so an ideal observer lands well inside [0.90, 0.99] — see §3 tolerance —
  with spread across coherences so commitment timing is non-degenerate.)
- **Ground-truth label = `sign(gamma)`** (the generative correct side), NOT the realized click sum, so
  the target is identical regardless of architecture and is the quantity the ideal observer estimates.
- Sizes: **6000 train / 2000 val / 4000 test** trials. Class-balanced.

### 1b. SECONDARY task (separate, pre-registered analysis) — delayed response
- `T = 40` bins: **evidence epoch** bins 1–30 (clicks as in 1a), **delay epoch** bins 31–40 with
  **`u_t = (0,0)` (zero input)**, response read at the trial end. Forces genuine memory across the
  zero-input delay; the post-commitment window therefore probes autonomy with no input to drive it.
  (Ideal final-bin accuracy ≈ 0.93.)
- Same coherence set, label, and sizes as 1a.

**Seeds.** Data generation seed is fixed per (task, split): `DATA_SEED = 20260630`. Network seeds:
`SEEDS = [0, 1, 2, 3, 4, 5]` (6 per architecture). All recorded.

---

## 2. Architectures — two families, three instances each, ≥6 seeds each

Every network is run **causally, one bin at a time**. At bin t it has seen `u_{≤t}` (within its
receptive field) and emits (i) a penultimate-layer hidden vector `h_t` and (ii) a per-bin decision
logit `o_t` from a linear read-out head. The per-bin hidden sequence `{h_t}` is the trajectory the
demix sees; `{o_t}` defines commitment. This per-bin causal unrolling is identical for all six
architectures — it is the only way the feedforward nets produce a task-time trajectory comparable to a
recurrent state, and it is the operationalization of the unfolding precondition.

**Recurrent family (state carried across bins):**
- `RNN` — vanilla Elman RNN (tanh).
- `GRU`.
- `LSTM`.

**Feedforward family (no state carried; `h_t` recomputed from a fixed input window ending at t):**
- `MLP` — multilayer perceptron over the length-`Lrf` window `u_{t−Lrf+1..t}` (left zero-padded), applied
  at every t.
- `TCN` — dilated causal temporal convolutional network; dilations chosen so the receptive field
  `Lrf ≥ T` (spans the whole task).
- `TFM` — small causal (masked) transformer encoder over `u_{1..t}`; `h_t` = output at position t.

**Receptive-field / unfolding precondition.** Each feedforward net's receptive field must span the task
(`Lrf ≥ T = 40`) so it *can* represent the full integral and match the I/O. A feedforward instance that
**cannot reach the accuracy tolerance (§3) is disqualified by the accuracy rule, never by its dynamics.**

**Parameter-budget rule (fixed now).** For each architecture, choose the hidden width (and for MLP/TCN/TFM
the depth as specified in code) so total trainable parameters fall in the band **[60k, 160k]**, and report
the actual count per architecture in the matched-accuracy table. Widths are selected by the deterministic
rule in `models.py::size_to_budget` (smallest width whose param count ≥ 60k, capped at 160k) — **not** tuned
to performance. `R = 8` read-out/PCA dimension is shared (below).

---

## 3. Matched accuracy (the unfolding precondition) — fixed BEFORE any demix

- Accuracy = **final-bin (t = T) decision accuracy on the held-out test set** (`o_T` sign vs label).
- **Inclusion rule:** a network is included in the dynamical analysis iff its test accuracy is **≥ 0.90**
  **and** within **±0.02** of the per-task mean test accuracy of all networks that are ≥ 0.90.
  Iterate once: compute mean over ≥0.90 nets, drop those outside ±0.02 of that mean, recompute the
  reported window. Networks failing this are **excluded by accuracy alone, before any dynamical measure
  is computed for them**, and the exclusion is logged.
- If an entire architecture fails to reach ≥0.90 across all seeds, record it as a deviation/addendum
  (the architecture could not meet the I/O precondition) and exclude it; do not relax the metric.
- Training: AdamW, fixed budget (max 60 epochs, early-stop on val, fixed LR schedule in `train.py`),
  per-bin cross-entropy against the trial label at **every** bin (standard trained-decision-net setup;
  applied identically to all architectures). Reported accuracy is final-bin only.

---

## 4. Trajectories and the input regressor (identical across all networks; matches the paper)

- **State reduction:** per network, fit PCA on the pooled test-set hidden vectors `h_t` (all bins, all
  test trials) and project to **`R = 8`** principal components (matches the brain pipeline `RANK = 8`).
  Recurrent nets use the recurrent hidden state; feedforward nets use the penultimate-layer activation.
- **Input regressor `u_t`:** the per-bin click-count vector `(n_left_t, n_right_t)` (`c = 2`), aligned to
  the trajectory, used **unchanged** as `U` — exactly as the brain pipeline passes `ev`. The demix's own
  delay embedding (`n_delays`) supplies input history identically for every network; **no extra manual lag
  is applied** (matching `analyze_commitment*.py`, which pass `ev` directly).
- **Demix instrument:** the deposited `~/cross-substrate-dynamics/src/demix.py`, imported **unchanged**.
  - `sdmdc_dom(X, U)` with **`rank = 8`, `n_delays = 2`** (the `analyze_commitment.py` settings).
  - `autonomy_index(X, U)` (pure-NumPy variance partition) with library defaults (`lam = 1e-2`, in-sample),
    matching `analyze_commitment_varpart.py`.
  - `flow_balance` computed by the paper's `flow_readouts` routine (copied verbatim from
    `analyze_commitment.py`) for the secondary.

---

## 5. Commitment definition (fixed now)

Per trial, the running decision variable is `p_t = sigmoid(o_t)` from the trained read-out applied at
each bin (the same head used for accuracy).

- **PRIMARY alignment (each network's OWN commitment):** commitment bin `t_c` = the first bin t at which
  `|p_t − 0.5| ≥ THETA` **and** `sign(p_τ − 0.5)` stays constant for all τ in `[t, T]` (crosses a threshold
  and *holds* to the trial end). `THETA = 0.40` (i.e. `p_t ≥ 0.90` or `≤ 0.10`). Trials with no such bin,
  or with `t_c` too late/early to admit both windows (§6), are excluded from that network's demix (logged).
- **ROBUSTNESS alignment (task-defined, identical across networks):** ideal-observer accumulator crossing —
  `t_c^IO` = first bin where the running cumulative log-likelihood ratio of the clicks (per-bin LLR under
  assumed coherence `IO_G0 = 0.10`) crosses a fixed bound `±B`, fixed in code as `B = 3.0` nats. This uses
  only the stimulus, so it is identical for all networks.

Both alignments are run; the **primary verdict uses the OWN-commitment alignment**; the ideal-observer
alignment is reported as robustness.

---

## 6. Demix windows around commitment (fixed now)

Width `W = 8` bins (matches brain `analyze_commitment.py`).
- **pre window** = the 8 bins immediately before commitment: `[t_c − 8, t_c)`.
- **post window** = the 8 bins immediately at/after commitment: `[t_c, t_c + 8)`.
A trial contributes to a network's demix iff both windows lie inside `[1, T]` (i.e. `9 ≤ t_c ≤ T − 8 = 32`).
For each network, trials are aligned to their own `t_c`, the pre and post windows are stacked across all
contributing trials, and the demix read-outs are computed on the stacked `(n_trials, 8, R)` tensors
(`X` = hidden PCs, `U` = click counts), exactly as the brain pipeline stacks per-session trials.
A network must have **≥ 20 contributing trials** in each window (brain `MIN_TR = 20`) or it is excluded
(logged). Identical for the ideal-observer alignment, using `t_c^IO`.

---

## 7. PRIMARY metric, hypothesis, and decision rule (FROZEN)

**Primary (lead) metric — the paper's commitment-attractor measure:**
`Δ sdmdc_dom = sdmdc_dom(post) − sdmdc_dom(pre)` per network (own-commitment alignment).

**Co-primary metric — the paper's autonomous-vs-input balance:** post-window
`auto_unique` and `r2_input` from `autonomy_index`, per network.

**Primary hypothesis (one-sided, pre-stated):** the recurrent family shows a **larger** commitment-aligned
`Δ sdmdc_dom` (movement toward a stronger autonomous attractor) **and higher** post-window `auto_unique`
than the feedforward family.

**Primary test.** Unit = network. Family = factor. For `Δ sdmdc_dom`:
**Mann–Whitney U, one-sided (recurrent > feedforward)** across all included networks (n_rec vs n_ff per
task), `α = 0.05`. Confirmatory: a linear mixed model
`Δ sdmdc_dom ~ family + (1 | architecture)` (architecture as random effect over the 6 instances), reporting
the family fixed-effect sign and p; the mixed model is corroborative and the Mann–Whitney is decisive.
The identical test is applied to the co-primary `auto_unique` (one-sided, recurrent > feedforward).

**KILL / CONFIRM (binary verdict rests on the LEAD metric `Δ sdmdc_dom`):**
- **CONFIRM** if the recurrent family exceeds the feedforward family on `Δ sdmdc_dom` at `α = 0.05`
  (Mann–Whitney one-sided), *and* this survives the triviality guard (§9). The co-primary `auto_unique`
  result is reported alongside: if it agrees → clean confirm; if it disagrees → **qualified confirm**,
  reported honestly as "attractor-strengthening separates the families, the variance-balance measure does
  not," with no spin.
- **KILL** if `Δ sdmdc_dom` does not separate the families in the predicted direction at `α = 0.05`, or
  reverses. A kill means: the paper's own dynamical instrument does **not** diagnose recurrent-vs-feedforward
  architecture under matched computation → the paper's implementation interpretation is not supported by a
  controlled test and **must be revised** (demote/retract that claim) per outcome (b) or (c) below.

The co-primary is **not** a toggle that can be swapped in to rescue a kill: the binary CONFIRM/KILL is
decided by `Δ sdmdc_dom` alone. `auto_unique` only qualifies the narration of a confirm.

---

## 8. SECONDARY metrics — exactly three, frozen, Holm-corrected, EXPLORATORY

Each one-sided, with its own pre-stated direction; Holm correction across the three; **all three reported
regardless of outcome.** A secondary separating while the primary kills is reported as "a different measure,
not the paper's, is the diagnostic one" — not as confirmation.

1. **Δ flow_balance** = `flow_balance(post) − flow_balance(pre)` (paper's autonomous-flow fraction).
   Direction: recurrent > feedforward.
2. **Post-commitment settling (speed collapse):** mean per-bin state speed in the post window,
   `speed = mean_t ||x_{t+1} − x_t||` (R-PC space). Direction: recurrent < feedforward (attractor → slowing).
3. **Autonomous-operator rollout generalization:** fit a delay-embedded (`n_delays = 2`, matching the
   embedding `sdmdc_dom` uses) autonomous linear operator `A` on the **pre**-window R-PC trajectories by
   least squares (input-free), then free-run it autonomously from each post trajectory's initial
   (delay-embedded) state over the post window and measure rollout MSE normalized by the post-trajectory
   variance. Direction: recurrent < feedforward (recurrent dynamics are more genuinely autonomous, so the
   input-free rollout generalizes better to held-out post data). Implemented in `analyze_cmc.py::rollout_mse`.
   (The R-PC-space operator is used rather than the N4SID internal latent because the DSA API does not expose
   latents for new/held-out data; the embedding and intent match `sdmdc_dom`.)

---

## 9. Triviality guards (pre-stated)

For every network, report the **post-window mean per-bin displacement** `disp = mean_t ||x_{t+1} − x_t||`
(same quantity as secondary 2) and the **post-window state variance** `var = mean trace(Cov(x_t))`, both in
R-PC space, plus the pre-window displacement.
- A network is flagged **trivially frozen** if its post-window mean displacement is below
  `EPS = 0.05 × (median pre-window displacement across all included networks of the same task)`.
- A primary separation that is **driven by one family freezing** (i.e. the separation disappears, or the
  flagged-frozen nets account for it, when frozen nets are removed) is **not** counted as a dynamical
  attractor; the verdict is downgraded and reported as degenerate. A genuine attractor must show a
  `sdmdc_dom` rise with **non-trivial** post displacement.

---

## 10. Outcomes (one is reported, whichever lands)

- **(a) Primary confirms** (and not trivial) → the paper's instrument diagnoses architecture under matched
  computation. Integrate as the computation-matched control that earns the implementation claim: add a
  Results subsection + figure; retire the "not computation-matched" limitation in the Discussion.
- **(b) Primary kills, ≥1 secondary separates (Holm)** → report both honestly; the paper must lead its
  model-side claim with the diagnostic measure and demote the non-diagnostic `Δ sdmdc_dom`; flag that the
  brain↔LLM result may need re-analysis with the diagnostic measure.
- **(c) Nothing separates** → the dynamical signature is not diagnostic of architecture under matched
  computation; the implementation interpretation is not supported; revise the paper to a measured
  cross-substrate *difference* without the recurrence/implementation mechanism; consider a dynamics/methods
  venue over the consciousness special issue.

`FINDINGS-CMC.md` will state the verdict, the matched-accuracy table, the primary result with test + CI,
all secondaries with Holm, the triviality diagnostics, and the integration action. No result is declared
settled from a single read; if it looks settled, it is re-run with fresh seeds (`SEEDS2 = [6,7,8,9,10,11]`)
before banking — that re-run is pre-authorized here and uses the identical frozen pipeline.

---

## 11. Operational freeze

- Seeds: `DATA_SEED = 20260630`; `SEEDS = [0..5]`; pre-authorized confirmatory `SEEDS2 = [6..11]`.
- BLAS threads pinned (`OMP_NUM_THREADS = OPENBLAS_NUM_THREADS = 1`) for the per-network demix loops.
- The demix instrument is `~/cross-substrate-dynamics/src/demix.py`, used **unchanged** (hash in FREEZE.txt).
- This file + `code/` (task.py, models.py, train.py, analyze_cmc.py, stats_cmc.py) are hashed into
  `FREEZE.txt`; demix is run only after FREEZE.txt exists. Any genuinely-forced deviation is recorded as a
  dated addendum at the bottom of this file; the metric and windows above are never silently changed.
