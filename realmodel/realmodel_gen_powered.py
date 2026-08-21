#!/usr/bin/env python3
"""Real-model generation-based commitment-attractor test — PROPERLY POWERED version.

Same design and instrument as realmodel_gen_attractor.py (strip the answer, generate, measure the
demix eigenvalue in a pre- vs post-commitment window around the committed answer token). Three changes,
all aimed only at statistical power (pre-registered in the PREREG amendment, frozen before results):
  1. Longer generation  : G = 48 continuation tokens (was 12).
  2. Wider windows       : W = 8 half-width -> 17-token windows (was 9), so the SubspaceDMDc eigenvalue
                           is estimated on a real trajectory rather than 4 tokens.
  3. t* = the position where the answer token actually appears within the first FIND generated tokens
     (was: strictly the first generated token). This recovers the many items where the model emits the
     answer at token 1-3 rather than token 0, raising usable-N without changing the metric.
No metric change; the read-out is the same post-minus-pre sdmdc_dom rise with item bootstrap. A NULL is
expected (feedforward, no post-commitment settling); a significant positive contractive rise would
refute the structural claim. Reported either way.

Usage:  DEMIX_DIR=. ITEMS=./construction_items.json \
        python3 realmodel_gen_powered.py <model_id> <out_json>
"""
import os, sys, json, warnings, time
warnings.filterwarnings("ignore")
import numpy as np

sys.path.insert(0, os.environ.get("DEMIX_DIR", os.path.dirname(os.path.abspath(__file__))))
from demix import autonomy_index, sdmdc_dom

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-1.5B-Instruct"
OUT   = sys.argv[2] if len(sys.argv) > 2 else "./realmodelpow_%s.json" % MODEL.split("/")[-1]
ITEMS = os.environ.get("ITEMS", "./construction_items.json")
DEPTHS = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
G = 48          # generated continuation tokens
W = 8           # window half-width (window = 2W+1 = 17 tokens)
FIND = 8        # search the answer token within the first FIND generated tokens
R = 8; RU = 8; NDELAYS = 1; SEED = 20260630; B = 1000
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
    items = [it for it in json.load(open(ITEMS)) if it["condition"] == "landmark"]

    raw = {d: [] for d in DEPTHS}
    n_committed = 0
    with torch.no_grad():
        for it in items:
            text = it["text"].rstrip(".").strip()
            ans = text.split()[-1]
            prompt = text[:len(text) - len(ans)].strip()
            if not prompt:
                continue
            atok = tok(" " + ans, add_special_tokens=False).input_ids
            if not atok:
                continue
            a0 = atok[0]
            enc = tok(prompt, return_tensors="pt").to(DEV)
            plen = enc.input_ids.shape[1]
            gen = model.generate(**enc, max_new_tokens=G, do_sample=False, pad_token_id=tok.eos_token_id)
            full = gen[0]
            genpart = full[plen:plen + FIND].tolist()
            if a0 not in genpart:                       # model did not commit to the answer early -> skip
                continue
            apos = plen + genpart.index(a0)             # position of the committed answer token
            if apos - W < 0 or apos + W >= full.shape[0]:
                continue
            n_committed += 1
            out = model(input_ids=full.unsqueeze(0))
            hs = out.hidden_states; nlay = len(hs) - 1
            emb = hs[0][0]
            win = list(range(apos - W, apos + W + 1))
            for d in DEPTHS:
                L = int(round(d * nlay))
                raw[d].append(dict(H=hs[L][0, win, :].float().cpu().numpy(),
                                   U0=emb[win, :].float().cpu().numpy(), item=it["id"]))
    del model
    if DEV == "cuda":
        torch.cuda.empty_cache()
    n_used = len(raw[DEPTHS[0]])
    print(f"[{MODEL}] dev={DEV} committed_items={n_committed}/{len(items)} extracted {time.time()-t0:.0f}s", flush=True)

    res = {"model": MODEL, "n_items": n_used, "G": G, "W": W, "FIND": FIND,
           "R": R, "RU": RU, "NDELAYS": NDELAYS, "seed": SEED, "depths": {}, "saved": {}}
    tstar = W

    for d in DEPTHS:
        entries = raw[d]
        if len(entries) < 10:
            continue
        muX, bX = pca_basis(np.concatenate([e["H"] for e in entries], 0), R)
        muU, bU = pca_basis(np.concatenate([e["U0"] for e in entries], 0), RU)
        X = np.stack([(e["H"] - muX) @ bX for e in entries])
        U = np.stack([(e["U0"] - muU) @ bU for e in entries])

        dom_full = sdmdc_dom(X, U, rank=R, nd=NDELAYS)
        ai = autonomy_index(X, U)

        def prepost(Xa, Ua):
            pre = sdmdc_dom(Xa[:, :tstar+1, :], Ua[:, :tstar+1, :], rank=R, nd=NDELAYS)
            post = sdmdc_dom(Xa[:, tstar:, :], Ua[:, tstar:, :], rank=R, nd=NDELAYS)
            return pre, post
        pre, post = prepost(X, U)
        ds = []
        for _ in range(B):
            bi = rng.integers(0, len(entries), len(entries))
            pr, po = prepost(X[bi], U[bi])
            if np.isfinite(pr) and np.isfinite(po):
                ds.append(po - pr)
        ds = np.array(ds)
        p2 = (2 * min((ds > 0).mean(), (ds < 0).mean())) if ds.size else None
        res["depths"][f"{d:.1f}"] = dict(
            n=len(entries), sdmdc_dom=rr(dom_full),
            contractive=bool(dom_full < 1.0) if np.isfinite(dom_full) else None,
            auto_unique=rr(ai["auto_unique"]), r2_input=rr(ai["r2_input"]),
            commit_pre=rr(pre), commit_post=rr(post),
            commit_delta=rr((post - pre) if (np.isfinite(pre) and np.isfinite(post)) else np.nan, 5),
            commit_delta_ci=[rr(np.percentile(ds, 2.5), 5), rr(np.percentile(ds, 97.5), 5)] if ds.size else None,
            commit_delta_p=rr(p2, 4))
        res["saved"][f"{d:.1f}"] = dict(X=np.round(X, 5).tolist(), U=np.round(U, 5).tolist())
        r = res["depths"][f"{d:.1f}"]
        print(f"  d={d:.1f}: sdmdc_dom={r['sdmdc_dom']} contractive={r['contractive']} "
              f"commit pre={r['commit_pre']} post={r['commit_post']} delta={r['commit_delta']} "
              f"p={r['commit_delta_p']} n={r['n']}", flush=True)

    json.dump(res, open(OUT, "w"), indent=1, default=str)
    print("wrote", OUT, "in", f"{time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
