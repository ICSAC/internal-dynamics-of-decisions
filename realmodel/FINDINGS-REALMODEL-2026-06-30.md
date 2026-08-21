# FINDINGS — real-model across-token commitment-attractor test (2026-06-30)

Pre-registered + frozen before results (`FREEZE.txt`); metric = the paper's own demix. Two designs;
the second (generation mode) is the one that can test the hypothesis.

## Prompt-token design (realmodel_across_token.py) — DEGENERATE, NOT USED
All 5 models: `frac_with_post = 0` (the answer commits ~2 tokens before the prompt ends, so no
post-commitment window exists) → the rise cannot be tested. Variance-partition levels were unstable on
12-token windows (Qwen-7B degenerated to auto_unique≈0.857 flat across depth, contradicting the
deposited balance). Conclusion: this design cannot test post-commitment settling; superseded.

## Generation-mode design (realmodel_gen_attractor.py) — THE TEST
Strip the answer, let the model generate answer + 12 continuation tokens, measure demix `sdmdc_dom`
in a pre-window (≤ committed answer) vs post-window (≥ committed answer); delta = the commitment-locked
attractor rise; item bootstrap B=1000. All 5 named models (Qwen2.5-1.5/3/7B, Llama-3.1-8B, gemma-2-9b),
6 depths each, 40 landmark items.

**RESULT = decision (a): NO commitment-locked attractor in any model.**
- **0 / 30** tests (5 models × 6 depths) show a significant positive commitment rise. Minimum p anywhere
  = 0.082. No model reproduces the brain's contractive commitment rise (brain: +0.016, p = 2.2×10⁻⁵,
  75/115 sessions).
- Where post-commitment dynamics move at all, dom|λ| goes **expansive (>1) in 10/30** depth-points —
  the opposite of settling into a contractive attractor.
- Direction matches the paper's structural claim on all five real models with a genuine post-commitment
  window and the same instrument.

**HONEST CAVEATS (keep in paper).**
- The eigenvalue estimator is **noisy on short generated windows** (2·WIN+1 = 9 tokens): per-depth delta
  point-estimates take both signs and bootstrap CIs are wide. This is corroboration of the *absence* of
  a commitment attractor, not a tight null.
- **answer-match is low** (5–13 / 40): these instruct models often do not emit the exact answer token
  first, so t* is the model's committed token whatever it is; the correct-only subset is small/noisier.
- The full-window `sdmdc_dom` regime level (mostly <1) and the auto_unique/r2_input balance are **not
  used** for any claim — short-window smoothness makes them unstable and, for the prompt-token design,
  they contradicted the deposited balance. The balance result lives in the dedicated balance analysis;
  this test speaks only to the commitment-locked *rise*.

## Bearing on the manuscript
Upgrades the model leg of the commitment comparison from "future work" to an executed, real-model,
time-locked test on all five named models: no commitment-locked attractor, paper-confirming direction,
reported with the noise caveat. The structural argument (a feedforward stack has no autonomous temporal
regime to settle into) remains the load-bearing claim; this is corroboration, not a sharpening.

Data: `results/realmodelgen_*.json` (per model: depths + saved reduced trajectories for CPU re-analysis).
Pod: RTX PRO 6000 Blackwell. Seed 20260630.

## Properly-powered follow-up (realmodel_gen_powered.py) — verdict (a) HOLDS
Power-only changes (pre-registered, frozen): G=48 tokens, W=8 (17-token windows), t* = where the answer
token appears within the first 8 generated tokens (raises usable-N). Landmark items, 5 models.
- **0 / 24 model-by-depth tests significant** (4 testable models × 6 depths; min p = 0.07). No commitment-
  locked attractor even when powered.
- **gemma-2-9b UNTESTABLE**: committed to its answer in only 6/40 generations (< 10 usable) → depths skipped.
- Weak, non-significant positive lean (20/24 deltas > 0) BUT the post-commitment eigenvalue drifts to the
  **marginal boundary (~1.0), not below it** into a contractive attractor; the brain's rise is within a
  clearly contractive regime (~0.90 → 0.93, well < 1). So even the lean is categorically not the brain's
  contractive-attractor signature.
- Conclusion unchanged and now backed by a powered run: no brain-like commitment attractor in the models;
  the wider windows rule out "it was just the short 9-token windows." Still corroboration (noisy estimator,
  n=14–16, gemma dropped), not a tight exclusion. Structural argument stays load-bearing.
Data: results/realmodelpow_*.json.
