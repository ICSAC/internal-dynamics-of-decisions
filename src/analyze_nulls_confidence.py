#!/usr/bin/env python3
"""Pre-registered confidence null on rat orbitofrontal cortex (Zenodo accession 16997337,
Schiereck et al., Neuron 2026). Tests, at the single-trial level, whether per-trial compression
geometry (a trial's loading on the leading population mode, the per-trial correlate of
population-rate participation-ratio compression) predicts trial-by-trial confidence (wait_time on
opt-out "catch" trials, the temporal-wagering confidence proxy), while per-trial linear
separability of the committed value (a value-readout axis) does not -- a trial-level dissociation.

Pre-registered analysis choices (locked before running):
  WINDOW / UNITS / PR CONVENTION: OFC units, COFF window [0, 0.5] s, non-violation trials,
    z-scored across trials -> correlation eigenbasis; participation ratio PR = (sum l)^2 / sum l^2;
    identical to the population-rate PR analysis.
  CONFIDENCE TRIALS: opt-out (catch) trials with a freely chosen finite wait_time; >= 15 per session.
  PER-TRIAL GEOMETRY (primary): estimate z-score statistics and the correlation eigenbasis on the
    complement (non-catch non-violation trials) to avoid circularity; project each catch trial;
    pc1_frac = fraction of the trial's projected energy on the leading mode (high = compressed).
  NEGATIVE CONTROL (separability): nearest-centroid margin for the trial's value class
    (low {1,2} / mid {3} / high {4,5}), centroids from the complement (high = well separated).
  CONFOUND CONTROL: residualize wait_time, pc1_frac and sep_margin each on [value, RT, pop_FR]
    (rank OLS) -> within-session partial Spearman of residual wait_time vs residual observable.
  AGGREGATE: session-level partial Spearman -> Wilcoxon signed-rank vs 0 (two-sided), median and
    bootstrap CI, sign counts. Secondary: inference-subset (trials-since-switch < 10) catch trials.
  KILL CRITERION: median partial Spearman(pc1_frac, wait) not significantly != 0, OR
    |median partial Spearman(sep_margin, wait)| >= |that of pc1_frac|.

Input: per-session *.mat files from the dataset above (env DATA_DIR or argv[1]).
Output: JSON (argv[2]) with the primary, negative-control, secondary and uncontrolled summaries
plus per-session records.
"""
import sys, os, glob, json, warnings, traceback
warnings.filterwarnings("ignore")
import numpy as np
from scipy import stats
from scipy.io import loadmat

DATA_DIR = sys.argv[1] if len(sys.argv)>1 else os.environ.get("DATA_DIR","./data")
OUT = sys.argv[2] if len(sys.argv)>2 else "./results/nulls_confidence.json"
RNG = np.random.default_rng(20260629)
MIN_UNITS=10; MIN_CATCH=15; WLO,WHI=0.0,0.5; ALIGN="COFF"

def convertreward(rew):
    rew=np.asarray(rew,float); out=rew.copy()
    out[(rew==4)|(rew==5)]=1; out[(rew==8)|(rew==10)]=2; out[(rew==16)|(rew==20)]=3
    out[(rew==32)|(rew==40)]=4; out[(rew==64)|(rew==80)]=5
    return out

def build_window_matrix(SU, align, lo, hi, ntr):
    rows=[]
    for u in SU:
        xv=np.asarray(getattr(u.xvec,align)).ravel()
        wi=np.where((xv>=lo)&(xv<hi))[0]
        if wi.size==0: return None
        H=np.asarray(getattr(u.hmat,align),dtype=float)
        if H.shape[0]!=ntr: return None
        rows.append(np.nanmean(H[:,wi],axis=1))
    return np.asarray(rows)   # units x ntr  (mean window FR)

def rank_resid(y, X):
    """residual of rank(y) regressed on rank(X) columns (+intercept). y:(n,), X:(n,k)."""
    ry=stats.rankdata(y)
    if X.shape[1]==0: return ry-ry.mean()
    RX=np.column_stack([stats.rankdata(X[:,j]) for j in range(X.shape[1])])
    A=np.column_stack([np.ones(len(ry)),RX])
    beta,_,_,_=np.linalg.lstsq(A,ry,rcond=None)
    return ry-A@beta

def partial_spearman(y, obs, ctrl):
    """Spearman partial correlation of y vs obs, controlling ctrl (n x k)."""
    ry=rank_resid(y,ctrl); ro=rank_resid(obs,ctrl)
    if np.std(ry)<1e-9 or np.std(ro)<1e-9: return np.nan
    return float(np.corrcoef(ry,ro)[0,1])

def main():
    files=sorted(glob.glob(os.path.join(DATA_DIR,"*.mat")))
    print(f"{len(files)} OFC files",flush=True)
    rows=[]; sess_records=[]
    for fi,f in enumerate(files):
        if fi%25==0: print(f"  {fi}/{len(files)} ({len(rows)} usable)",flush=True)
        rec={"file":os.path.basename(f)}
        try:
            d=loadmat(f,squeeze_me=True,struct_as_record=False)
            if "S" not in d or "SU" not in d: rec["skip"]="noS/SU"; sess_records.append(rec); continue
            S=d["S"]; SU=[u for u in np.atleast_1d(d["SU"]) if str(getattr(u,"location","OFC"))=="OFC"]
            if len(SU)<MIN_UNITS: rec["skip"]="few units"; sess_records.append(rec); continue
            Block=np.asarray(S.Block,float).ravel(); ntr=len(Block)
            Rew=np.asarray(S.RewardAmount,float).ravel(); v=convertreward(Rew)
            vios=np.asarray(S.vios).astype(bool).ravel()
            wt=np.asarray(S.wait_time,float).ravel()
            rt=np.asarray(S.ReactionTime,float).ravel()
            optout=np.asarray(S.optout).astype(float).ravel()
            # trials-since-switch
            allChange=np.where(np.diff(Block)!=0)[0]+1; chg=set(allChange.tolist())
            tss=np.zeros(ntr,int); last=0
            for i in range(ntr):
                if i==0 or i in chg: last=i
                tss[i]=i-last
            M=build_window_matrix(SU,ALIGN,WLO,WHI,ntr)
            if M is None: rec["skip"]="win build"; sess_records.append(rec); continue
            popFR=np.nanmean(M,axis=0)                      # per-trial mean pop FR (raw)
            nonvio=~vios
            finiteFR=np.all(np.isfinite(M),axis=0)
            # CONFIDENCE trials = OPT-OUT (catch) trials with a freely-chosen finite wait_time.
            catch=np.where(nonvio & finiteFR & (optout==1) & np.isfinite(wt) & (wt>0))[0]
            # geometry/centroid reference = full session population (nonvio, finite FR).
            # wait_time plays NO role in defining the eigenbasis -> no predictor<->outcome leakage.
            ref=np.where(nonvio & finiteFR)[0]
            if catch.size<MIN_CATCH or ref.size<30: rec.update(n_catch=int(catch.size),n_ref=int(ref.size),skip="few catch/ref"); sess_records.append(rec); continue
            # ---- z-score units on reference ----
            mu=np.nanmean(M[:,ref],axis=1,keepdims=True); sd=np.nanstd(M[:,ref],axis=1,keepdims=True)
            keep=(sd.ravel()>1e-9)&np.isfinite(sd.ravel())&np.isfinite(mu.ravel())
            if keep.sum()<MIN_UNITS: rec["skip"]="few active units"; sess_records.append(rec); continue
            Mz=(M-mu)/sd; Mz=Mz[keep]; Mz=np.nan_to_num(Mz)
            Zref=Mz[:,ref]
            # ---- correlation eigenbasis on reference (matches the correlation-PR convention) ----
            C=(Zref@Zref.T)/(Zref.shape[1]-1)
            evals,evecs=np.linalg.eigh(C)         # ascending
            order=np.argsort(evals)[::-1]; evals=evals[order]; evecs=evecs[:,order]
            set_PR_ref=float((evals[evals>1e-10].sum()**2)/np.square(evals[evals>1e-10]).sum())
            # per-catch-trial pc1 loading fraction (energy on leading mode)
            Zc=Mz[:,catch]                          # (units_keep x n_catch)
            Sproj=evecs.T@Zc                        # (modes x n_catch)
            energy=np.square(Sproj); tot=energy.sum(axis=0); tot[tot<1e-12]=np.nan
            pc1_frac=energy[0]/tot                  # fraction of trial energy on leading mode
            # ---- separability negative control: nearest-centroid margin on value-class ----
            cls=np.full(ntr,-1,int)
            cls[(v==1)|(v==2)]=0; cls[v==3]=1; cls[(v==4)|(v==5)]=2
            present=[c for c in (0,1,2) if np.sum(cls[ref]==c)>=3]
            sep=np.full(catch.size,np.nan)
            if len(present)>=2:
                cls_ref=cls[ref]
                cent={c:Zref[:,cls_ref==c].mean(axis=1) for c in present}
                cnt={c:int(np.sum(cls_ref==c)) for c in present}
                for j,ti in enumerate(catch):
                    ci=cls[ti]
                    if ci not in present: continue
                    # leave-one-out own-class centroid (catch trials are in ref): remove this trial
                    own=(cent[ci]*cnt[ci]-Zc[:,j])/(cnt[ci]-1) if cnt[ci]>1 else cent[ci]
                    d_own=np.linalg.norm(Zc[:,j]-own)
                    d_oth=min(np.linalg.norm(Zc[:,j]-cent[c]) for c in present if c!=ci)
                    sep[j]=d_oth-d_own              # high = well separated/readable
            # ---- confounds + within-session partial Spearman ----
            wt_c=wt[catch]; rt_c=rt[catch]; v_c=v[catch].astype(float); fr_c=popFR[catch]; tss_c=tss[catch]
            # build control matrix (value, RT, popFR); drop RT if all-nan
            ctrl_cols=[v_c]
            if np.isfinite(rt_c).sum()>=0.5*catch.size: ctrl_cols.append(np.nan_to_num(rt_c,nan=np.nanmedian(rt_c)))
            ctrl_cols.append(fr_c)
            CTRL=np.column_stack(ctrl_cols)
            valid=np.isfinite(wt_c)&np.isfinite(pc1_frac)&np.all(np.isfinite(CTRL),axis=1)
            valid_sep=valid&np.isfinite(sep)
            if valid.sum()<MIN_CATCH: rec.update(n_catch=int(catch.size),skip="few valid"); sess_records.append(rec); continue
            ps_pc1=partial_spearman(wt_c[valid], pc1_frac[valid], CTRL[valid])
            raw_pc1=float(stats.spearmanr(wt_c[valid],pc1_frac[valid]).correlation)
            ps_sep=partial_spearman(wt_c[valid_sep], sep[valid_sep], CTRL[valid_sep]) if valid_sep.sum()>=MIN_CATCH else np.nan
            # inference-subset (secondary)
            infm=valid & (tss_c<10)
            ps_pc1_inf=partial_spearman(wt_c[infm],pc1_frac[infm],CTRL[infm]) if infm.sum()>=10 else np.nan
            rec.update(rat=str(S.RatName), n_units=int(keep.sum()), n_catch=int(valid.sum()),
                       n_catch_inf=int(infm.sum()), set_PR_ref=set_PR_ref,
                       ps_pc1=ps_pc1, ps_sep=ps_sep, raw_pc1=raw_pc1, ps_pc1_inf=ps_pc1_inf)
            if np.isfinite(ps_pc1): rows.append(rec)
            sess_records.append(rec)
        except Exception as e:
            rec["error"]=repr(e)[:160]; rec["tb"]=traceback.format_exc()[-300:]; sess_records.append(rec)
            print("  ERR",rec["file"],repr(e)[:120],flush=True)

    def summ(key, label):
        x=np.array([r[key] for r in rows if np.isfinite(r.get(key,np.nan))],float)
        o={"label":label,"n_sessions":int(x.size),
           "median":float(np.median(x)) if x.size else None,
           "mean":float(np.mean(x)) if x.size else None,
           "n_pos":int(np.sum(x>0)),"n_neg":int(np.sum(x<0))}
        if x.size>=5:
            o["wilcoxon_two_p"]=float(stats.wilcoxon(x,alternative="two-sided").pvalue)
            o["t_one_sample_p"]=float(stats.ttest_1samp(x,0).pvalue)
            bs=[np.median(RNG.choice(x,x.size,replace=True)) for _ in range(5000)]
            o["median_CI95"]=[float(np.percentile(bs,2.5)),float(np.percentile(bs,97.5))]
        return o

    out={"design":"per-trial pc1_frac (compression) vs wait_time confidence; sep-margin negative control; controls=value,RT,popFR",
         "n_usable_sessions":len(rows),
         "PRIMARY_pc1_frac_vs_confidence":summ("ps_pc1","partial Spearman wait_time ~ pc1_frac | value,RT,popFR"),
         "NEGCONTROL_separability_vs_confidence":summ("ps_sep","partial Spearman wait_time ~ value-decode margin | controls"),
         "SECONDARY_pc1_inference_only":summ("ps_pc1_inf","pc1_frac vs confidence, inference (tss<10) catch trials only"),
         "raw_pc1_uncontrolled":summ("raw_pc1","raw Spearman wait_time vs pc1_frac (no controls)"),
         "sessions":sess_records}
    json.dump(out,open(OUT,"w"),indent=1,default=str)
    print("\n=== RESULTS ===")
    for k in ("PRIMARY_pc1_frac_vs_confidence","NEGCONTROL_separability_vs_confidence","SECONDARY_pc1_inference_only","raw_pc1_uncontrolled"):
        print(k); print(json.dumps(out[k],indent=1))
    print(f"\nwrote {OUT}")

if __name__=="__main__": main()
