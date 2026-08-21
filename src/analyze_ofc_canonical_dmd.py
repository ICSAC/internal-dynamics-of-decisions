#!/usr/bin/env python3
"""Robustness check for the rat orbitofrontal cortex recurrence test using a canonical
delay-embedded dynamic mode decomposition (DSA toolbox), in place of a single-step proxy.

Value-matched inference (trial-since-switch < 10) versus control (trial-since-switch >= 20),
averaged over NDRAW subsamples, reporting the dominant |eigenvalue| of the delay-embedded DMD
operator A_v for n_delays in {1, 3}.

Input: the per-session PC-trajectory pickle from extract_ofc.py (argv[1], default
./data/ofc_pctraj.pkl), built from the rat OFC value-based decision dataset (Zenodo accession
16997337; Schiereck et al., Neuron 2026).

Output: a JSON list (argv[2], default ./results/dsa_canonical_validate.json) of per-session
dominant-eigenvalue differences for each n_delays; aggregate statistics are computed separately
from this JSON.
"""
import sys, os, json, pickle
import numpy as np
from DSA import DMD
PKL=sys.argv[1] if len(sys.argv)>1 else "./data/ofc_pctraj.pkl"
OUT=sys.argv[2] if len(sys.argv)>2 else "./results/dsa_canonical_validate.json"
NDRAW=20; MIN_TRIALS=8; SEED=20260630; NDELAYS=(1,3)

def level_match(inf,ctl,v,rng):
    li=v[inf]; lc=v[ctl]; si=[];sc=[]
    for lv in np.unique(np.concatenate([li,lc])):
        ii=inf[li==lv]; cc=ctl[lc==lv]; m=min(len(ii),len(cc))
        if m==0: continue
        si.append(rng.choice(ii,m,replace=False)); sc.append(rng.choice(cc,m,replace=False))
    if not si: return None,None
    return np.concatenate(si),np.concatenate(sc)

def dom_lambda(traj_idx, pct, nd):
    # pct[i]: (R, bins) -> need (ntrials, bins, R)
    X=np.stack([pct[i].T for i in traj_idx],axis=0)   # ntrials x bins x R
    R=X.shape[2]
    try:
        d=DMD(X, n_delays=nd, rank=R*nd, device="cpu", verbose=False)
        d.fit()
        ev=np.linalg.eigvals(np.asarray(d.A_v))
        return float(np.max(np.abs(ev)))
    except Exception:
        return np.nan

def main():
    data=pickle.load(open(PKL,"rb"))
    res=[]
    for fi,(fn,S) in enumerate(data.items()):
        if fi%30==0: print(f"  {fi}/{len(data)}",flush=True)
        try:
            pct=S["pctraj"].astype(np.float64); tss=np.asarray(S["tss"])
            vios=np.asarray(S["vios"]).astype(bool); v=np.asarray(S["v"]); nonvio=~vios
            inf=np.where(nonvio&(tss<10))[0]; ctl=np.where(nonvio&(tss>=20))[0]
            if inf.size<MIN_TRIALS or ctl.size<MIN_TRIALS: continue
            rng=np.random.default_rng(abs(hash(fn))%(2**31)+SEED)
            acc={nd:{"i":[],"c":[]} for nd in NDELAYS}
            for _ in range(NDRAW):
                si,sc=level_match(inf,ctl,v,rng)
                if si is None or len(si)<MIN_TRIALS: continue
                for nd in NDELAYS:
                    di=dom_lambda(si,pct,nd); dc=dom_lambda(sc,pct,nd)
                    if np.isfinite(di) and np.isfinite(dc): acc[nd]["i"].append(di); acc[nd]["c"].append(dc)
            rec={"file":fn,"rat":S.get("rat")}
            ok=True
            for nd in NDELAYS:
                if len(acc[nd]["i"])<5: ok=False; break
                rec[f"dom_inf_nd{nd}"]=float(np.mean(acc[nd]["i"]))
                rec[f"dom_ctl_nd{nd}"]=float(np.mean(acc[nd]["c"]))
                rec[f"d_dom_nd{nd}"]=float(np.mean(acc[nd]["i"])-np.mean(acc[nd]["c"]))
            if ok: res.append(rec)
        except Exception as e:
            print("  ERR",fn,repr(e)[:100],flush=True)
    os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
    json.dump(res,open(OUT,"w"),indent=1,default=str)
    for nd in NDELAYS:
        dd=np.array([r[f"d_dom_nd{nd}"] for r in res])
        print(f"n_delays={nd}: n={len(dd)} d_dom median={np.median(dd):+.4f} mean={np.mean(dd):+.4f} pos={int((dd>0).sum())} neg={int((dd<0).sum())}")
    print(f"wrote {OUT}")
if __name__=="__main__": main()
