# Pre-registration — real-model across-token commitment-attractor test

**Frozen before any real-model result is inspected.** This is the clean version of the paper's
weaker model leg: it replaces the noisy *across-layer* eigenvalue proxy (current Fig. 5a) with a
*within-pass temporal* analogue (across token positions), on the real instruction-tuned models,
multi-seed, with the SAME instrument the brain analysis uses. Frozen because the outcome can
**refute** the paper's structural claim (see Decision rule); nothing below may change once a real
activation is read.

## Why the across-token axis is the right analogue
A transformer has no recurrence across depth, so the across-layer trajectory (used in the current
draft) is not a dynamical system — norm/projection artefacts inflate its eigenvalue at late layers.
The across-token sequence at a fixed depth *is* an ordered, within-pass sequence and is the closest
temporal analogue of a population trajectory. The brain result to mirror: SubspaceDMDc dom|λ|(A_v)
**rises after commitment, commitment-locked** (+0.016, p = 2.2×10⁻⁵, 75/115 sessions).

## Substrates / models (all five, frozen)
Qwen2.5-1.5B-Instruct, Qwen2.5-3B-Instruct, Qwen2.5-7B-Instruct, Llama-3.1-8B-Instruct,
Gemma-2-9b-it. Item set: the 40 deposited `landmark` inference items in `construction_items.json`
(the same inference condition used for the across-layer leg). **Multi-seed robustness comes from the
five models × item bootstrap (B = 1000), not from authored paraphrases** — no new items are
fabricated. Per-model and pooled-across-models statistics are both reported.

## Trajectory definition (frozen)
For each item and each fractional depth d ∈ {0.4, 0.5, 0.6, 0.7, 0.8, 0.9} of the stack:
- Take the hidden state at layer ⌊d·L⌋ for the final **K = 16** token positions of the prompt →
  a (K × H) matrix. (K = 16, window width 4, and the ±offsets below are the frozen script
  constants in `realmodel_across_token.py`, hash-pinned in FREEZE.txt.)
- Reduce to **R = 8** dimensions by PCA fit per (model, depth) over all items (same R as the brain
  OFC/EC analyses). This is the across-token trajectory **x**_t, t = token index.
- Input regressor **u**_t = the layer-0 input embeddings at the same K positions (the InputDSA drive),
  identical to the balance analysis.

## Commitment analog across tokens (frozen)
Per item, the **commit position** t\* = the first of the final-K token positions at which the logit
lens at that position and depth ranks the correct answer token top and it stays top through the last
position. Items with no such stable t\* in the final-K window are recorded and excluded from the
commitment-locked test (counted, reported — no silent drop). A `fixed`/stimulus-style alignment
(align to the last prompt token instead of t\*) is the control alignment, mirroring the brain's
stimulus-aligned comparison.

## Instrument and read-outs (frozen — the paper's own)
`demix.py` (SubspaceDMDc / N4SID), UNCHANGED, hash-pinned in FREEZE.txt. Read-outs:
- **Primary:** `sdmdc_dom` = |dominant eigenvalue of A_v|, computed in sliding token windows
  (width 4) aligned to t\*. Test statistic = post-commit minus pre-commit dom|λ|
  (post window centred +2 tokens, pre window centred −2 tokens), per item, aggregated across items
  by Wilcoxon signed-rank; commitment-aligned rise compared with the control-aligned rise by paired
  Wilcoxon (exactly the brain test).
- **Secondary:** `auto_unique` = R²(x'|x,u) − R²(x'|u) over the same windows (robust cross-substrate
  read-out); and the absolute regime level (dom|λ| ≷ 1: contractive attractor vs expansive/feedforward).

## Multi-seed (frozen)
- **B = 1000 item-bootstrap** resamples for CIs on every reported statistic.
- Robustness across the **five models** is the replication axis (a real effect must hold across
  families/scales, not one model).
- The GPU pass **saves the reduced (K×R) across-token trajectories, the input (K×RU) regressor, and
  t\* per item** so every windowed read-out is recomputable on CPU without re-extraction (correcting a
  prior run that saved only summary stats).
- No other degrees of freedom. No metric added or swapped after seeing a result. NDELAYS = 1 for the
  short across-token sequences (matches the across-layer commit script).

## Decision rule (both directions pre-stated — this is the anti-rigging safeguard)
- **(a) CONFIRMS the paper (predicted):** across tokens the model shows **no** commitment-locked
  contractive-attractor rise — either dom|λ| stays ≥ 1 / expansive with no t\*-aligned rise, or the
  post−pre rise is ≤ 0 or not significant, or the commitment-aligned rise does not exceed the
  control-aligned rise (paired test NS). Then the current structural claim is *empirically* backed on
  the real models and Fig. 5a is upgraded from the across-layer proxy to this within-pass measure.
- **(b) REFUTES the paper:** a properly powered t\*-aligned analysis shows a **contractive
  commitment-locked dom|λ| rise** analogous to the brain (post > pre, sig, exceeding the control
  alignment, in a majority of items in ≥ 3 of 5 models, CI excluding 0). Then the model DOES build a
  commitment attractor within the pass → the structural "feedforward stack has no attractor" reading
  is wrong → the dissociation is rewritten and this becomes the headline correction. **Report it
  either way.**
- **(c) AMBIGUOUS:** mixed across models / underpowered → report as such; the across-layer proxy
  stays and the leg remains the weaker half, but now with the stronger measure attempted and shown.

## Compute
Single GPU pod (bf16, `output_hidden_states`, eager attention). Est. all five models × ≥120 items ×
3 variants × 6 depths ≈ tens of minutes on an RTX-6000-class pod (the demix itself is cheap CPU).
`extract` on the pod → pull result JSONs from the compute node. HF token from env (gated Llama/Gemma).

## Provenance
Script `realmodel_across_token.py` + this pre-reg + the frozen item/variant set are hashed in
`FREEZE.txt` before the first real run. Seed = 20260630 for all resampling/PCA.

---

## Amendment (same session, before any generation-mode result) — GENERATION MODE

The prompt-token design above was run first and is **structurally unable** to test post-commitment
settling: on these factual statements the answer commits ~2 tokens before the prompt ends, so no item
has a post-commitment window (frac_with_post = 0 across all models). Its variance-partition levels
were also unstable on 12-token windows (Qwen-7B degenerated to auto_unique≈0.857 flat across depth,
contradicting the paper's own deposited balance for that model), so the prompt-token levels are NOT
used. This limitation was identified from the RUN STRUCTURE (frac_post and a reconciliation against
the deposited balance), not from any attractor rise/no-rise being observed — no post-commitment rise
statistic could be computed at all — so amending the design here is not metric-hunting.

**Generation-mode design (frozen; script `realmodel_gen_attractor.py`, hashed in FREEZE.txt):**
strip the answer word from each `landmark` statement; the model greedily generates the answer plus
G = 12 continuation tokens; a single forward pass over prompt+generation gives the across-token
trajectory. t* = the position of the first generated token (the model's committed answer). Window =
[t*-WIN .. t*+WIN], WIN = 4, so a genuine post-commitment window exists by construction. Same
instrument, same R = 8 / RU = 8 PCA, NDELAYS = 1. Read-out: set-level sdmdc_dom in a pre-window
(≤ t*) vs a post-window (≥ t*); delta = post − pre = the commitment-locked attractor rise; item
bootstrap B = 1000 for CI and two-sided p; also computed on the correct-answer subset. auto_unique /
r2_input reported as secondary (short-window smoothness caveat retained). Answer-match rate reported.

**Decision rule (generation mode):** (a) CONFIRMS = no significant positive post-minus-pre rise
across ≥3 of 5 models (delta ≤ 0 or CI includes 0), i.e. no commitment-locked attractor; (b) REFUTES
= a significant positive rise in ≥3 of 5 models (contractive dom|λ| rising after commitment like the
brain) → the model builds a commitment attractor → rewrite the dissociation; (c) AMBIGUOUS = mixed /
underpowered → the across-layer structural leg stands, the across-token attempt is reported honestly
as inconclusive. Report whichever way it lands.

---

## Amendment 2 (before any powered result) — PROPERLY-POWERED generation run

The generation-mode run returned a null (0/30 significant) but was noisy on 9-token windows with low
answer-match, so it corroborated rather than sharpened. The manuscript names "a properly powered
version (longer generations, more items)" as the residual; this run is that pre-stated step, NOT a
re-test hunting a different answer. Frozen params (`realmodel_gen_powered.py`, hashed in FREEZE.txt),
POWER-ONLY changes, metric unchanged:
- G = 48 generated tokens (was 12); window half-width W = 8 → 17-token windows (was 9), so the
  eigenvalue is fit on a real trajectory.
- t* = the position where the answer token first appears within the first FIND = 8 generated tokens
  (was: strictly token 0), recovering items where the model emits the answer at token 1–3 and raising
  usable-N from the 40 landmark items.
- Same demix, same R=8/RU=8 PCA, same pre-vs-post sdmdc_dom rise, same item bootstrap B=1000.
Decision rule unchanged: (a) no significant positive rise across ≥3/5 models = no attractor (predicted);
(b) significant positive contractive rise ≥3/5 = refute; (c) still mixed/noisy = report as such and keep
the structural leg. Report whichever way it lands; the null already on record is not to be overturned by
tuning.
