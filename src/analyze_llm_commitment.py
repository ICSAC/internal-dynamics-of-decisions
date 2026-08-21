#!/usr/bin/env python3
"""Across-layer autonomous-attractor analogue for a transformer language model, aligned
to the logit-lens commit layer.

For each inference ("landmark") item in the construction-task set
(data/construction_items.json), the script runs a forward pass through a Hugging Face
causal language model, takes the hidden state at the position that predicts the answer
token, and reads its across-layer trajectory (the residual-stream state as a function of
network depth, unit-normalized per layer to track direction rather than residual-norm
growth). A logit-lens pass defines the model's "commit layer" L* as the first layer at
which the answer token becomes the top next-token prediction. The dominant eigenvalue
magnitude of an autonomous dynamic model fit in sliding LAYER windows is then computed
and aligned to L*, and compared with windows at fixed fractional depths. This is the
language-model counterpart to the commitment-locked autonomous-recurrence analysis of the
cortical decision data: it asks whether an across-depth attractor strengthens at the
layer where the model commits to its answer.

Input : a Hugging Face model id (argv[1]); construction items at
        $DATA_DIR/construction_items.json (default ./data).
Output: JSON (argv[2], default ./results/llm_commit_<model>.json) with, per L*-aligned and
        fixed-depth layer offset, the across-layer dominant-eigenvalue magnitude and a
        bootstrap confidence interval, plus the median commit layer.

Models: Hugging Face Qwen2.5 / Llama-3.1 / Gemma-2. Gated checkpoints are read with a
token supplied via the HF_TOKEN environment variable.
"""
import os, sys, json, warnings, time
warnings.filterwarnings("ignore")
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-7B-Instruct"
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join("results", "llm_commit_%s.json" % MODEL.split("/")[-1])
ITEMS = os.path.join(os.environ.get("DATA_DIR", "./data"), "construction_items.json")
HF_TOKEN = os.environ.get("HF_TOKEN")          # set for gated checkpoints
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
R = 8; WL = 4; NDELAYS = 1; SEED = 20260630
rng = np.random.default_rng(SEED)


def dmd_dom(traj_stack, nd=NDELAYS, ridge=1e-2):
    """plain autonomous DMD dom|lambda| on across-layer trajectories (n_items, Wlayers, R)."""
    P1, P2 = [], []
    for Z in traj_stack:
        T, Rr = Z.shape
        if T < nd + 2:
            continue
        H = np.concatenate([Z[nd-1-k:T-k] for k in range(nd)], axis=1)
        P1.append(H[:-1]); P2.append(H[1:])
    if not P1:
        return np.nan
    X1 = np.concatenate(P1, 0); X2 = np.concatenate(P2, 0); d = X1.shape[1]
    A = np.linalg.solve(X1.T @ X1 + ridge*np.eye(d), X1.T @ X2).T
    return float(np.max(np.abs(np.linalg.eigvals(A))))


def main():
    tok = AutoTokenizer.from_pretrained(MODEL, token=HF_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map=DEVICE,
                                                 output_hidden_states=True, attn_implementation="eager",
                                                 token=HF_TOKEN).eval()
    # unembedding for the logit lens
    W_U = model.get_output_embeddings().weight        # (vocab, hidden)
    norm = model.model.norm if hasattr(model, "model") and hasattr(model.model, "norm") else None
    items = [it for it in json.load(open(ITEMS)) if it["condition"] == "landmark"]
    t0 = time.time()
    per_item_traj = []      # (n_layers, hidden) at the answer-predicting position
    commit_layers = []
    with torch.no_grad():
        for it in items:
            text = it["text"]; ans = text.rstrip(".").strip().split()[-1]
            enc = tok(text, return_tensors="pt").to(DEVICE)
            ids = enc.input_ids[0]
            atok = tok(" " + ans, add_special_tokens=False).input_ids
            if not atok:
                continue
            a0 = atok[0]
            # find the answer first-token position in ids (last occurrence)
            pos = (ids == a0).nonzero(as_tuple=True)[0]
            if len(pos) == 0:
                continue
            ap = int(pos[-1].item())
            cp = ap - 1                              # position predicting the answer token
            if cp < 1:
                continue
            hs = model(**enc).hidden_states          # tuple (nlay+1) of (1,seq,hid)
            nlay = len(hs)
            traj = np.stack([hs[L][0, cp, :].float().cpu().numpy() for L in range(nlay)], 0)  # (nlay, hid)
            traj = traj / (np.linalg.norm(traj, axis=1, keepdims=True) + 1e-9)   # unit-norm per layer: directional dynamics, de-confound residual-norm growth
            # logit lens: identify the first layer whose top prediction is the answer token
            commitL = nlay - 1
            for L in range(nlay):
                h = hs[L][0, cp, :]
                if norm is not None:
                    h = norm(h)
                logit = (h.float() @ W_U.float().T)
                if int(torch.argmax(logit).item()) == a0:
                    commitL = L; break
            per_item_traj.append(traj); commit_layers.append(commitL)
    del model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    nlay = per_item_traj[0].shape[0]
    print(f"[{MODEL}] {nlay} layers, {len(per_item_traj)} items, extracted {time.time()-t0:.0f}s; "
          f"commit layer L* median={int(np.median(commit_layers))} range[{min(commit_layers)},{max(commit_layers)}]", flush=True)
    # PCA on all items x layers
    allflat = np.concatenate([t for t in per_item_traj], 0)
    mu = allflat.mean(0); U, S, Vt = np.linalg.svd(allflat - mu, full_matrices=False); basis = Vt[:R].T
    trajs = [((t - mu) @ basis) for t in per_item_traj]    # each (nlay, R)
    Lstar = np.array(commit_layers)

    # dom|lambda| in sliding layer windows, aligned to L* (offset in layers), pooled across items
    offsets = list(range(-8, 9))
    res = {"model": MODEL, "n_layers": nlay, "n_items": len(trajs),
           "commit_layer_median": int(np.median(Lstar)), "WL": WL, "Lstar_aligned": [], "fixed_aligned": []}
    for off in offsets:
        stack = []
        for tr, ls in zip(trajs, Lstar):
            s = ls + off; e = s + WL
            if s < 0 or e > nlay:
                continue
            stack.append(tr[s:e, :])
        if len(stack) >= 10:
            dom = dmd_dom(stack)
            # bootstrap
            ds = [dmd_dom([stack[i] for i in rng.integers(0, len(stack), len(stack))]) for _ in range(150)]
            ds = np.array([x for x in ds if np.isfinite(x)])
            res["Lstar_aligned"].append(dict(offset=off, dom=round(dom, 4) if np.isfinite(dom) else None,
                                              ci=[round(float(np.percentile(ds, 2.5)), 4), round(float(np.percentile(ds, 97.5)), 4)] if ds.size else None,
                                              n=len(stack)))
    # fixed (depth-fraction) aligned control: windows at fixed layer positions
    for frac in [i/16 for i in range(2, 15)]:
        L0 = int(frac * (nlay - WL))
        stack = [tr[L0:L0+WL, :] for tr in trajs if L0+WL <= nlay]
        if len(stack) >= 10:
            res["fixed_aligned"].append(dict(layer=L0, frac=round(frac, 2), dom=round(dmd_dom(stack), 4), n=len(stack)))
    os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=1, default=str)
    print("=== L*-ALIGNED across-layer dom|lambda| (offset 0 = commit layer) ===", flush=True)
    for r in res["Lstar_aligned"]:
        if r["dom"] is not None:
            print(f"  L*{r['offset']:+d}: dom={r['dom']:.4f} CI{r['ci']} n={r['n']}", flush=True)
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
