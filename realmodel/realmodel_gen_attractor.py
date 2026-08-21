#!/usr/bin/env python3
"""Real-model GENERATION-based across-token commitment-attractor test (GPU pod).

Faithful within-pass temporal analogue of the brain's post-commitment settling. The prompt-token
version (realmodel_across_token.py) is structurally unable to test post-commitment dynamics because
the answer is committed ~2 tokens before the prompt ends (no post-commitment window). Here the answer
is stripped from each prompt, the model GENERATES the answer plus a continuation, and the across-token
trajectory is measured in a window centred on the committed answer token so a genuine post-commitment
window exists (frac_post = 1 by construction).

Brain result to mirror: dom|lambda|(A_v) is contractive and RISES after commitment (settles into an
attractor). Prediction for a feedforward transformer: no post-commitment rise (each generated token is
a fresh forward pass; there is no autonomous temporal regime that settles). Reported either way.

Same instrument as the brain / balance analyses (demix.py: SubspaceDMDc sdmdc_dom + variance-partition
auto_unique / r2_input). Set-level read-outs on batched (n_items, T, R) trajectories, with item
bootstrap (B) for CIs and a two-sided bootstrap p on the post-minus-pre rise. Reduced trajectories and
the committed-answer index are saved for CPU re-analysis. Design frozen in
PREREG-REALMODEL-ACROSSTOKEN.md (generation-mode section) and hashed in FREEZE.txt.

Usage:  DEMIX_DIR=. ITEMS=./construction_items.json \
        python3 realmodel_gen_attractor.py <model_id> <out_json>
Requires demix.py and the DSA package; HF token in ~/.cache/huggingface/token for gated models.
"""
import os, sys, json, warnings, time
warnings.filterwarnings("ignore")
import numpy as np

sys.path.insert(0, os.environ.get("DEMIX_DIR", os.path.dirname(os.path.abspath(__file__))))
from demix import autonomy_index, sdmdc_dom            # the paper's frozen instrument

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-1.5B-Instruct"
OUT   = sys.argv[2] if len(sys.argv) > 2 else "./realmodelgen_%s.json" % MODEL.split("/")[-1]
ITEMS = os.environ.get("ITEMS", "./construction_items.json")
DEPTHS = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
G = 12          # generated continuation tokens (post-commitment window source)
WIN = 4         # commitment-locked window half-width (window = [t*-WIN .. t*+WIN], t* = answer pos)
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
    n_correct = 0
    with torch.no_grad():
        for it in items:
            text = it["text"].rstrip(".").strip()
            ans = text.split()[-1]
            prompt = text[:len(text) - len(ans)].strip()          # statement minus the answer word
            if not prompt:
                continue
            atok = tok(" " + ans, add_special_tokens=False).input_ids
            a0 = atok[0] if atok else None
            enc = tok(prompt, return_tensors="pt").to(DEV)
            plen = enc.input_ids.shape[1]
            gen = model.generate(**enc, max_new_tokens=G, do_sample=False,
                                  pad_token_id=tok.eos_token_id)
            full = gen[0]                                          # (plen + G,)
            apos = plen                                           # first generated token = committed answer
            if apos - WIN < 0 or apos + WIN >= full.shape[0]:
                continue
            correct = int(full[apos].item() == a0) if a0 is not None else 0
            n_correct += correct
            out = model(input_ids=full.unsqueeze(0))
            hs = out.hidden_states; nlay = len(hs) - 1
            emb = hs[0][0]
            win = list(range(apos - WIN, apos + WIN + 1))         # 2*WIN+1 positions, t* at index WIN
            for d in DEPTHS:
                L = int(round(d * nlay))
                raw[d].append(dict(H=hs[L][0, win, :].float().cpu().numpy(),
                                   U0=emb[win, :].float().cpu().numpy(),
                                   correct=correct, item=it["id"]))
    del model
    if DEV == "cuda":
        torch.cuda.empty_cache()
    n_used = len(raw[DEPTHS[0]])
    print(f"[{MODEL}] dev={DEV} n_items={n_used} answer_match={n_correct}/{n_used} "
          f"extracted {time.time()-t0:.0f}s", flush=True)

    res = {"model": MODEL, "n_items": n_used, "answer_match": n_correct, "G": G, "WIN": WIN,
           "R": R, "RU": RU, "NDELAYS": NDELAYS, "seed": SEED, "depths": {}, "saved": {}}
    tstar = WIN                                                    # aligned by construction

    for d in DEPTHS:
        entries = raw[d]
        if len(entries) < 10:
            continue
        muX, bX = pca_basis(np.concatenate([e["H"] for e in entries], 0), R)
        muU, bU = pca_basis(np.concatenate([e["U0"] for e in entries], 0), RU)
        X = np.stack([(e["H"] - muX) @ bX for e in entries])      # (n, 2WIN+1, R)
        U = np.stack([(e["U0"] - muU) @ bU for e in entries])
        cor = np.array([e["correct"] for e in entries], bool)

        dom_full = sdmdc_dom(X, U, rank=R, nd=NDELAYS)
        ai = autonomy_index(X, U)

        def prepost(Xa, Ua):
            pre = sdmdc_dom(Xa[:, :tstar+1, :], Ua[:, :tstar+1, :], rank=R, nd=NDELAYS)     # up to & incl. commit
            post = sdmdc_dom(Xa[:, tstar:, :], Ua[:, tstar:, :], rank=R, nd=NDELAYS)         # commit onward
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
        # correct-answer-only subset (model committed to the right answer)
        corr_delta = None
        if cor.sum() >= 10:
            prc, poc = prepost(X[cor], U[cor])
            corr_delta = rr((poc - prc) if (np.isfinite(prc) and np.isfinite(poc)) else np.nan, 5)

        res["depths"][f"{d:.1f}"] = dict(
            n=len(entries), sdmdc_dom=rr(dom_full),
            contractive=bool(dom_full < 1.0) if np.isfinite(dom_full) else None,
            auto_unique=rr(ai["auto_unique"]), r2_input=rr(ai["r2_input"]),
            commit_pre=rr(pre), commit_post=rr(post),
            commit_delta=rr((post - pre) if (np.isfinite(pre) and np.isfinite(post)) else np.nan, 5),
            commit_delta_ci=[rr(np.percentile(ds, 2.5), 5), rr(np.percentile(ds, 97.5), 5)] if ds.size else None,
            commit_delta_p=rr(p2, 4), commit_delta_correct_only=corr_delta)
        res["saved"][f"{d:.1f}"] = dict(X=np.round(X, 5).tolist(), U=np.round(U, 5).tolist(),
                                        correct=[int(c) for c in cor])
        r = res["depths"][f"{d:.1f}"]
        print(f"  d={d:.1f}: sdmdc_dom={r['sdmdc_dom']} contractive={r['contractive']} "
              f"auto_u={r['auto_unique']} r2_in={r['r2_input']} | commit pre={r['commit_pre']} "
              f"post={r['commit_post']} delta={r['commit_delta']} p={r['commit_delta_p']}", flush=True)

    json.dump(res, open(OUT, "w"), indent=1, default=str)
    print("wrote", OUT, "in", f"{time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
