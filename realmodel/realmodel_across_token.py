#!/usr/bin/env python3
"""Real-model across-token commitment-attractor test (GPU pod).

The within-pass TEMPORAL analogue that upgrades the paper's weaker model leg from the noisy
across-LAYER proxy to an across-TOKEN measure, using the SAME instrument as the brain analysis
(SubspaceDMDc / demix.py). Frozen design in PREREG-REALMODEL-ACROSSTOKEN.md; hashed in FREEZE.txt.

Brain result to mirror: dom|lambda|(A_v) is CONTRACTIVE (~0.90-0.93 < 1) and RISES at behavioural
commitment, commitment-locked. Prediction for a feedforward transformer: the across-token operator is
NOT a contractive attractor, and there is no commitment-locked rise (the model commits at the end of
the token sequence, leaving no post-commitment settling window).

For each landmark inference item, at each fractional depth d, the across-token trajectory over the
final K token positions at layer round(d*L) is x_t (token index = time); the input u_t is the
layer-0 input embedding at the same positions (the InputDSA drive). t* = the commitment position =
the logit-lens position at which the answer token first becomes the top-1 prediction and stays (the
answer-predicting position cp is recorded for reference). Read-outs, computed at the SET level with
the instrument's batched contract (X: (n_items, T, R), U: (n_items, T, RU)):
  - regime: sdmdc_dom (contractive attractor if <1) and auto_unique / r2_input over the full window;
  - commitment-locked: set-level sdmdc_dom in a pre-window vs a post-window of a t*-aligned stack,
    delta = post - pre, with item bootstrap (B) for the CI and the two-sided bootstrap p; plus
    frac_with_post = fraction of items that even have a post-commitment window.

The reduced (K x R) trajectory, the (K x RU) input, and t* are saved per item so any windowed
read-out is recomputable on CPU without re-extraction.

Usage:  HF_TOKEN=... DEMIX_DIR=. ITEMS=./construction_items.json \
        python3 realmodel_across_token.py <model_id> <out_json>
Requires demix.py (the paper's instrument) and the DSA package.
"""
import os, sys, json, warnings, time
warnings.filterwarnings("ignore")
import numpy as np

sys.path.insert(0, os.environ.get("DEMIX_DIR", os.path.dirname(os.path.abspath(__file__))))
from demix import autonomy_index, sdmdc_dom            # the paper's frozen instrument

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-1.5B-Instruct"
OUT   = sys.argv[2] if len(sys.argv) > 2 else "./realmodel_%s.json" % MODEL.split("/")[-1]
ITEMS = os.environ.get("ITEMS", "./construction_items.json")
DEPTHS = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
K = 12          # fixed final-token window forming the across-token trajectory
R = 8           # state subspace dim
RU = 8          # input subspace dim
NDELAYS = 1     # short sequences
WIN = 3         # commitment-locked window width (pre = t*-WIN..t*, post = t*..t*+WIN)
SEED = 20260630
B = 1000        # item bootstrap
rng = np.random.default_rng(SEED)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
DT = torch.bfloat16 if DEV == "cuda" else torch.float32


def pca_basis(pooled, r):
    mu = pooled.mean(0)
    _, _, Vt = np.linalg.svd(pooled - mu, full_matrices=False)
    return mu, Vt[:r].T


def rr(x, n=4):
    return round(float(x), n) if (x is not None and np.isfinite(x)) else None


def main():
    t0 = time.time()
    hf = os.environ.get("HF_TOKEN")
    tok = AutoTokenizer.from_pretrained(MODEL, token=hf)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=DT, device_map=DEV, output_hidden_states=True,
        attn_implementation="eager", token=hf).eval()
    W_U = model.get_output_embeddings().weight
    norm = getattr(getattr(model, "model", model), "norm", None)
    items = [it for it in json.load(open(ITEMS)) if it["condition"] == "landmark"]

    raw = {d: [] for d in DEPTHS}
    n_seen = 0
    with torch.no_grad():
        for it in items:
            text = it["text"]; ans = text.rstrip(".").strip().split()[-1]
            enc = tok(text, return_tensors="pt").to(DEV)
            ids = enc.input_ids[0]
            atok = tok(" " + ans, add_special_tokens=False).input_ids
            if not atok:
                continue
            a0 = atok[0]
            pos = (ids == a0).nonzero(as_tuple=True)[0]
            if len(pos) == 0:
                continue
            ap = int(pos[-1].item()); cp = ap - 1
            seq = ids.shape[0]
            if cp < 1 or seq < K:
                continue
            win = list(range(seq - K, seq))
            if cp not in win:
                continue
            n_seen += 1
            tcp = win.index(cp)
            out = model(**enc); hs = out.hidden_states; nlay = len(hs) - 1
            emb = hs[0][0]
            Lc = int(round(0.6 * nlay))
            tstar = tcp                      # default: answer-predicting position
            for j, p in enumerate(win):
                h = hs[Lc][0, p, :]
                if norm is not None:
                    h = norm(h)
                if int(torch.argmax(h.float() @ W_U.float().T).item()) == a0:
                    tstar = j; break         # first logit-lens commit position within the window
            for d in DEPTHS:
                L = int(round(d * nlay))
                raw[d].append(dict(H=hs[L][0, win, :].float().cpu().numpy(),
                                   U0=emb[win, :].float().cpu().numpy(),
                                   tstar=int(tstar), tcp=int(tcp), item=it["id"]))
    del model
    if DEV == "cuda":
        torch.cuda.empty_cache()
    n_used = len(raw[DEPTHS[0]])
    print(f"[{MODEL}] dev={DEV} n_items={n_used} extracted {time.time()-t0:.0f}s", flush=True)

    res = {"model": MODEL, "n_items": n_used, "K": K, "R": R, "RU": RU, "NDELAYS": NDELAYS,
           "WIN": WIN, "seed": SEED, "depths": {}, "saved": {}}

    for d in DEPTHS:
        entries = raw[d]
        if len(entries) < 10:
            continue
        muX, bX = pca_basis(np.concatenate([e["H"] for e in entries], 0), R)
        muU, bU = pca_basis(np.concatenate([e["U0"] for e in entries], 0), RU)
        X = np.stack([(e["H"] - muX) @ bX for e in entries])     # (n, K, R)
        U = np.stack([(e["U0"] - muU) @ bU for e in entries])    # (n, K, RU)
        ts = np.array([e["tstar"] for e in entries])
        tcp = np.array([e["tcp"] for e in entries])

        # --- regime (set level, full window) ---
        dom_full = sdmdc_dom(X, U, rank=R, nd=NDELAYS)
        ai = autonomy_index(X, U)

        # --- commitment-locked rise (t*-aligned stack, set level, item bootstrap) ---
        keep = [i for i in range(len(entries)) if ts[i] - WIN >= 0 and ts[i] + WIN <= K - 1]
        commit = None
        if len(keep) >= 10:
            A = np.stack([X[i, ts[i]-WIN:ts[i]+WIN+1, :] for i in keep])    # (m, 2WIN+1, R)
            Au = np.stack([U[i, ts[i]-WIN:ts[i]+WIN+1, :] for i in keep])

            def prepost(Xa, Ua):
                pre = sdmdc_dom(Xa[:, :WIN+1, :], Ua[:, :WIN+1, :], rank=R, nd=NDELAYS)
                post = sdmdc_dom(Xa[:, WIN:, :], Ua[:, WIN:, :], rank=R, nd=NDELAYS)
                return pre, post
            pre, post = prepost(A, Au)
            ds = []
            for _ in range(B):
                bi = rng.integers(0, len(keep), len(keep))
                pr, po = prepost(A[bi], Au[bi])
                if np.isfinite(pr) and np.isfinite(po):
                    ds.append(po - pr)
            ds = np.array(ds)
            p2 = (2 * min((ds > 0).mean(), (ds < 0).mean())) if ds.size else None
            commit = dict(pre=rr(pre), post=rr(post),
                          delta=rr((post - pre) if (np.isfinite(pre) and np.isfinite(post)) else np.nan, 5),
                          ci=[rr(np.percentile(ds, 2.5), 5), rr(np.percentile(ds, 97.5), 5)] if ds.size else None,
                          p_two=rr(p2, 4), m=len(keep), frac_with_post=round(len(keep)/len(entries), 3))

        res["depths"][f"{d:.1f}"] = dict(
            n=len(entries), sdmdc_dom=rr(dom_full),
            contractive=bool(dom_full < 1.0) if np.isfinite(dom_full) else None,
            auto_unique=rr(ai["auto_unique"]), r2_input=rr(ai["r2_input"]),
            commit=commit, tstar_median=int(np.median(ts)), tcp_median=int(np.median(tcp)))
        res["saved"][f"{d:.1f}"] = dict(X=np.round(X, 5).tolist(), U=np.round(U, 5).tolist(),
                                        tstar=[int(x) for x in ts], tcp=[int(x) for x in tcp])
        r = res["depths"][f"{d:.1f}"]
        cd = r["commit"]["delta"] if r["commit"] else None
        cp_p = r["commit"]["p_two"] if r["commit"] else None
        fp = r["commit"]["frac_with_post"] if r["commit"] else 0.0
        print(f"  d={d:.1f}: sdmdc_dom={r['sdmdc_dom']} contractive={r['contractive']} "
              f"auto_u={r['auto_unique']} r2_in={r['r2_input']} | commit d={cd} p={cp_p} "
              f"frac_post={fp} tstar_med={r['tstar_median']}/{K}", flush=True)

    json.dump(res, open(OUT, "w"), indent=1, default=str)
    print("wrote", OUT, "in", f"{time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
