#!/usr/bin/env python3
"""Reduce each rat orbitofrontal cortex recording session to a low-dimensional population
trajectory in PC space, for downstream recurrent-versus-input dynamics analysis.

Input: a directory (the DATA_DIR environment variable, or argv[1]) of per-session MATLAB .mat
files from the rat OFC value-based decision dataset (Zenodo accession 16997337; Schiereck et al.,
Neuron 2026). Each file provides single-unit peri-event histograms (hmat) aligned to choice
offset (COFF), trial block structure, offered reward amount, and violation flags.

For each session with at least MIN_UNITS OFC units, units are z-scored across non-violation
trials, the top-R principal components of the unit covariance are taken, and every trial is
projected onto that basis to give a (trials x R x bins) trajectory tensor over the [W0, W1] s
window. Trial-since-switch (tss) is computed from block changes.

Output: a single pickle (argv[2], default ./data/ofc_pctraj.pkl) mapping each session filename to
a dict with the PC trajectory (float32), trial-since-switch, violation flags, reward level,
subject id, unit count, and bin count.
"""
import sys, os, glob, pickle, warnings
warnings.filterwarnings("ignore")
import numpy as np, scipy.io as sio

# DATA_DIR should point at the directory containing the OFC session .mat files.
DATA_DIR = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("DATA_DIR", "./data")
OUT = sys.argv[2] if len(sys.argv) > 2 else "./data/ofc_pctraj.pkl"
MIN_UNITS=12; MIN_BINS=8; R=8; ALIGN="COFF"; W0,W1=0.0,1.0

def convertreward(rew):
    rew=np.asarray(rew,float); out=rew.copy()
    out[(rew==4)|(rew==5)]=1; out[(rew==8)|(rew==10)]=2; out[(rew==16)|(rew==20)]=3
    out[(rew==32)|(rew==40)]=4; out[(rew==64)|(rew==80)]=5
    return out

def traj_tensor(SU,lo,hi,ntr):
    xv0=np.asarray(getattr(SU[0].xvec,ALIGN)).ravel()
    wi=np.where((xv0>=lo)&(xv0<hi))[0]
    if wi.size<MIN_BINS: wi=np.where((xv0>=-0.25)&(xv0<1.25))[0]
    if wi.size<MIN_BINS: return None,0
    cube=[]
    for u in SU:
        xv=np.asarray(getattr(u.xvec,ALIGN)).ravel()
        if xv.shape[0]!=xv0.shape[0]: return None,0
        H=np.asarray(getattr(u.hmat,ALIGN),dtype=float)
        if H.shape[0]!=ntr: return None,0
        cube.append(H[:,wi])
    return np.stack(cube,axis=1), wi.size   # trials x units x bins

files=sorted(glob.glob(os.path.join(DATA_DIR,"*.mat")))
print(f"{len(files)} files",flush=True); out={}
for fi,f in enumerate(files):
    if fi%25==0: print(f"  {fi}/{len(files)} ({len(out)} saved)",flush=True)
    try:
        d=sio.loadmat(f,squeeze_me=True,struct_as_record=False)
        if "S" not in d or "SU" not in d: continue
        S=d["S"]; SU=[u for u in np.atleast_1d(d["SU"]) if str(getattr(u,"location","OFC"))=="OFC"]
        if len(SU)<MIN_UNITS: continue
        Block=np.asarray(S.Block,float).ravel(); ntr=len(Block)
        v=convertreward(np.asarray(S.RewardAmount,float).ravel())
        vios=np.asarray(S.vios).astype(bool).ravel()
        allChange=np.where(np.diff(Block)!=0)[0]+1; chg=set(allChange.tolist())
        tss=np.zeros(ntr,int); last=0
        for i in range(ntr):
            if i==0 or i in chg: last=i
            tss[i]=i-last
        T,nb=traj_tensor(SU,W0,W1,ntr)
        if T is None: continue
        nonvio=~vios; allidx=np.where(nonvio)[0]
        flat=T[allidx].transpose(1,0,2).reshape(T.shape[1],-1)
        mu=np.nanmean(flat,axis=1); sd=np.nanstd(flat,axis=1)
        keep=(sd>1e-9)&np.isfinite(sd)&np.isfinite(mu)
        if keep.sum()<MIN_UNITS: continue
        Tz=(T-mu.reshape(1,-1,1))/sd.reshape(1,-1,1); Tz=Tz[:,keep,:]; Tz=np.nan_to_num(Tz)
        U_=Tz[allidx].transpose(1,0,2).reshape(int(keep.sum()),-1)
        C=np.cov(U_); ev,evec=np.linalg.eigh(C); basis=evec[:,::-1][:,:R]   # units x R
        pctraj=np.einsum('ur,tub->trb', basis, Tz).astype(np.float32)        # trials x R x bins
        out[os.path.basename(f)]=dict(pctraj=pctraj, tss=tss.astype(np.int16),
            vios=vios, v=v.astype(np.int8), rat=str(S.RatName), n_units=int(keep.sum()), nbins=int(nb))
    except Exception as e:
        print("  ERR",os.path.basename(f),repr(e)[:100],flush=True)
os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
pickle.dump(out,open(OUT,"wb"))
print(f"saved {len(out)} sessions -> {OUT} ({os.path.getsize(OUT)/1e6:.1f} MB)",flush=True)
