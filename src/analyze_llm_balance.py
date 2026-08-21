#!/usr/bin/env python3
"""Recurrent-versus-input balance and inference-versus-control recruitment for a
transformer language model (reference reimplementation of the GPU analysis).

NOTE: The original activation-extraction script ran on a GPU and was not retained. This
is a faithful reference reimplementation built from the analysis design and the shared
instrument in demix.py. It reproduces the structure and quantities of the per-model
result files (inputdsa_model_<model>.json); exact numbers can differ slightly from the
original run because of tokenizer-dependent item filtering and GPU/CPU numerics.

Design. For a Hugging Face causal language model, each construction-task item
(data/construction_items.json) is run through a forward pass with all hidden states
captured. Items with condition 'landmark' are the inference set; 'var_filler' are the
length-/content-matched control. For each item the across-token trajectory over the final
K=12 token positions is taken at SIX fractional network depths (0.4, 0.5, 0.6, 0.7, 0.8,
0.9 of the layer stack); the exogenous input U is the layer-0 input-embedding trajectory
at the same K positions. Per-item trajectories are reduced to R=8 dimensions (states) and
RU=8 dimensions (input) by PCA fit on the pooled positions across both conditions, and the
two demix read-outs are computed per condition with demix.autonomy_index (variance
partition: auto_unique = R2(x'|x,u) - R2(x'|u), the recurrence share; r2_input = R2(x'|u),
the input-driven share) and demix.sdmdc_dom (autonomous dominant-eigenvalue magnitude).
The inference-minus-control difference in auto_unique is bootstrapped over items.

Input : a Hugging Face model id (argv[1]); construction items at
        $DATA_DIR/construction_items.json (default ./data).
Output: JSON (argv[2], default ./results/inputdsa_model_<model>.json) with, at each
        fractional depth, auto_unique / r2_input / r2_full / sdmdc_dom for inference and
        control, the inference-control difference in auto_unique with a bootstrap 95% CI
        and the bootstrap fraction positive; and a summary block with r2_input_mean,
        auto_unique_mean, d_auto_unique_mean, d_auto_unique_commitL (the deepest probe)
        and n_layers_pos (number of depths with a positive recruitment).

Models: Hugging Face Qwen2.5 / Llama-3.1 / Gemma-2. Gated checkpoints are read with a
token supplied via the HF_TOKEN environment variable.
"""
import os, sys, json, warnings, time
warnings.filterwarnings("ignore")
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from demix import autonomy_index, sdmdc_dom

MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-7B-Instruct"
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join("results", "inputdsa_model_%s.json" % MODEL.split("/")[-1])
ITEMS = os.path.join(os.environ.get("DATA_DIR", "./data"), "construction_items.json")
HF_TOKEN = os.environ.get("HF_TOKEN")          # set for gated checkpoints
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

K = 12                                          # final token positions in each trajectory
R = 8                                           # state subspace dimension
RU = 8                                          # input subspace dimension
NDELAYS = 3                                     # delay embedding for SubspaceDMDc
FRACS = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]          # fractional network depths probed
NBOOT = 200                                     # item bootstrap draws for the difference CI
SEED = 20260630
rng = np.random.default_rng(SEED)


def pca_fit(rows, k):
    """Top-k PCA basis from pooled (N, H) rows; returns (mean, basis (H,k))."""
    M = np.concatenate(rows, 0)
    mu = M.mean(0)
    _, _, Vt = np.linalg.svd(M - mu, full_matrices=False)
    return mu, Vt[:k].T


def project(arr, mu, basis):
    """Project a (K, H) trajectory into the PCA subspace -> (K, k)."""
    return (arr - mu) @ basis


def extract(model, tok):
    """Forward pass per item; return inference/control dicts of per-item layer slices.
    Each dict maps an item index to {'emb': (K,H), L: (K,H) for L in probe layers}."""
    n_hs = model.config.num_hidden_layers + 1                  # hidden_states length
    lidx = [int(f * (n_hs - 1)) for f in FRACS]
    items = json.load(open(ITEMS))
    cond_map = {"landmark": "inf", "var_filler": "ctl"}
    out = {"inf": [], "ctl": []}
    t0 = time.time()
    with torch.no_grad():
        for it in items:
            tag = cond_map.get(it["condition"])
            if tag is None:
                continue
            enc = tok(it["text"], return_tensors="pt").to(DEVICE)
            if enc.input_ids.shape[1] < K:                    # need K final positions
                continue
            hs = model(**enc).hidden_states                    # tuple (n_hs) of (1,seq,hid)
            rec = {"emb": hs[0][0, -K:, :].float().cpu().numpy()}
            for L in lidx:
                rec[L] = hs[L][0, -K:, :].float().cpu().numpy()
            out[tag].append(rec)
    print(f"[{MODEL}] {n_hs} hidden states, probe layers {lidx}; "
          f"inf={len(out['inf'])} ctl={len(out['ctl'])}; extracted {time.time()-t0:.0f}s", flush=True)
    return out, n_hs, lidx


def condition_arrays(recs, key, mu, basis):
    """Stack per-item projected trajectories for one condition -> (n, K, dim)."""
    return np.stack([project(r[key], mu, basis) for r in recs], 0)


def main():
    tok = AutoTokenizer.from_pretrained(MODEL, token=HF_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=DTYPE, device_map=DEVICE,
                                                 output_hidden_states=True, attn_implementation="eager",
                                                 token=HF_TOKEN).eval()
    data, n_hs, lidx = extract(model, tok)
    del model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    inf, ctl = data["inf"], data["ctl"]
    n_inf, n_ctl = len(inf), len(ctl)

    # input subspace: PCA on pooled layer-0 embeddings across both conditions
    emb_mu, emb_basis = pca_fit([r["emb"] for r in inf + ctl], RU)
    U_inf = condition_arrays(inf, "emb", emb_mu, emb_basis)     # (n_inf, K, RU)
    U_ctl = condition_arrays(ctl, "emb", emb_mu, emb_basis)

    res = {"model": MODEL, "n_layers": n_hs, "K": K, "R": R, "RU": RU,
           "n_delays": NDELAYS, "n_inf": n_inf, "n_ctl": n_ctl, "layers": {}}

    for frac, L in zip(FRACS, lidx):
        x_mu, x_basis = pca_fit([r[L] for r in inf + ctl], R)
        Xi = condition_arrays(inf, L, x_mu, x_basis)           # (n_inf, K, R)
        Xc = condition_arrays(ctl, L, x_mu, x_basis)           # (n_ctl, K, R)
        ai = autonomy_index(Xi, U_inf); ac = autonomy_index(Xc, U_ctl)
        d_auto = ai["auto_unique"] - ac["auto_unique"]
        # item bootstrap of the inference-control auto_unique difference
        boot = []
        for _ in range(NBOOT):
            bi = rng.integers(0, n_inf, n_inf); bc = rng.integers(0, n_ctl, n_ctl)
            da = autonomy_index(Xi[bi], U_inf[bi])["auto_unique"] - \
                 autonomy_index(Xc[bc], U_ctl[bc])["auto_unique"]
            if np.isfinite(da):
                boot.append(da)
        boot = np.array(boot)
        res["layers"]["frac%.1f" % frac] = dict(
            L=int(L),
            auto_unique_inf=round(ai["auto_unique"], 4), auto_unique_ctl=round(ac["auto_unique"], 4),
            d_auto_unique=round(d_auto, 4),
            d_auto_CI95=[round(float(np.percentile(boot, 2.5)), 4),
                         round(float(np.percentile(boot, 97.5)), 4)] if boot.size else None,
            frac_boot_pos=round(float(np.mean(boot > 0)), 4) if boot.size else None,
            r2_input_inf=round(ai["r2_input"], 4), r2_input_ctl=round(ac["r2_input"], 4),
            d_r2_input=round(ai["r2_input"] - ac["r2_input"], 4),
            r2_full_inf=round(ai["r2_full"], 4), r2_full_ctl=round(ac["r2_full"], 4),
            sdmdc_dom_inf=round(sdmdc_dom(Xi, U_inf, rank=R, nd=NDELAYS), 4),
            sdmdc_dom_ctl=round(sdmdc_dom(Xc, U_ctl, rank=R, nd=NDELAYS), 4),
        )
        print(f"  frac{frac:.1f} (L{L}): auto_unique inf={ai['auto_unique']:.4f} ctl={ac['auto_unique']:.4f} "
              f"d={d_auto:+.4f}  r2_input inf={ai['r2_input']:.4f}", flush=True)

    # summary aggregates across the six depth probes
    lay = list(res["layers"].values())
    r2_in_all = [v["r2_input_inf"] for v in lay] + [v["r2_input_ctl"] for v in lay]
    auto_all = [v["auto_unique_inf"] for v in lay] + [v["auto_unique_ctl"] for v in lay]
    d_auto = [v["d_auto_unique"] for v in lay]
    res["summary"] = dict(
        r2_input_mean=round(float(np.mean(r2_in_all)), 4),
        auto_unique_mean=round(float(np.mean(auto_all)), 4),
        d_auto_unique_mean=round(float(np.mean(d_auto)), 4),
        d_auto_unique_commitL=lay[-1]["d_auto_unique"],       # deepest probe (frac 0.9)
        n_layers_pos=int(np.sum(np.array(d_auto) > 0)),
        n_probes=len(FRACS),
    )
    os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=1, default=str)
    print("summary:", json.dumps(res["summary"]), flush=True)
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
