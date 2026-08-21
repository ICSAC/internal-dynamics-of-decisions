#!/usr/bin/env python3
"""Frozen statistics + verdict (PREREG-CMC.md sec.3,7,8,9,10). RUN AFTER analyze_cmc.py.

Applies the accuracy-inclusion rule, the one-sided family tests, Holm correction for the three
secondaries, the triviality guard, and maps to verdict (a)/(b)/(c). Writes results/stats_<task>.json.
"""
import os, sys, json
import numpy as np
from scipy import stats

RES = os.environ.get("CMC_RES", "./results")
ALPHA = 0.05


def boot_diff_ci(a, b, n=5000, seed=0):
    """bootstrap CI of mean(a)-mean(b)."""
    rng = np.random.default_rng(seed)
    a = np.asarray(a); b = np.asarray(b)
    d = [rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(n)]
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def mwu(rec, ff, greater=True):
    """one-sided Mann-Whitney; greater=True tests rec>ff, else rec<ff."""
    rec = np.array([x for x in rec if np.isfinite(x)])
    ff = np.array([x for x in ff if np.isfinite(x)])
    if len(rec) < 3 or len(ff) < 3:
        return dict(p=None, U=None, n_rec=len(rec), n_ff=len(ff),
                    med_rec=None, med_ff=None, ci=None)
    alt = "greater" if greater else "less"
    U, p = stats.mannwhitneyu(rec, ff, alternative=alt)
    lo, hi = boot_diff_ci(rec, ff)
    return dict(p=float(p), U=float(U), n_rec=len(rec), n_ff=len(ff),
                med_rec=float(np.median(rec)), med_ff=float(np.median(ff)),
                mean_rec=float(np.mean(rec)), mean_ff=float(np.mean(ff)),
                ci_mean_diff=[round(lo, 4), round(hi, 4)])


def holm(pvals):
    """Holm-Bonferroni; returns adjusted p (same order)."""
    idx = np.argsort([p if p is not None else 1.0 for p in pvals])
    m = len(pvals); adj = [None] * m; running = 0.0
    for rank, i in enumerate(idx):
        p = pvals[i]
        if p is None:
            adj[i] = None; continue
        a = (m - rank) * p
        running = max(running, a)
        adj[i] = min(1.0, running)
    return adj


def get(rows, fam, key, align="own"):
    return [r[align].get(key, np.nan) for r in rows if r["family"] == fam and align in r]


def apply_inclusion(rows):
    """PREREG sec.3: acc>=0.90 AND within +/-0.02 of mean of the >=0.90 set (one iteration)."""
    accs = {(r["arch"], r["seed"]): r["test_acc"] for r in rows}
    ge90 = [a for a in accs.values() if a >= 0.90]
    if not ge90:
        for r in rows:
            r["included"] = False
        return rows, dict(n_ge90=0, mean=None, lo=None, hi=None)
    mean = float(np.mean(ge90))
    lo, hi = mean - 0.02, mean + 0.02
    for r in rows:
        r["included"] = bool(r["test_acc"] >= 0.90 and lo <= r["test_acc"] <= hi)
    return rows, dict(n_ge90=len(ge90), mean=round(mean, 4), lo=round(lo, 4), hi=round(hi, 4))


def main():
    task = sys.argv[1] if len(sys.argv) > 1 else "primary"
    rows = json.load(open(os.path.join(RES, f"analysis_{task}.json")))
    # numeric coercion (json dumped with default=str for nan)
    for r in rows:
        for tag in ("own", "io"):
            if tag in r:
                for k, v in list(r[tag].items()):
                    try:
                        r[tag][k] = float(v)
                    except (TypeError, ValueError):
                        r[tag][k] = np.nan
    rows, incl = apply_inclusion(rows)
    inc = [r for r in rows if r["included"]]
    excl = [(r["arch"], r["seed"], round(r["test_acc"], 4)) for r in rows if not r["included"]]

    def split(key, align="own"):
        rec = [r[align].get(key, np.nan) for r in inc if r["family"] == "recurrent"]
        ff = [r[align].get(key, np.nan) for r in inc if r["family"] == "feedforward"]
        return rec, ff

    # ---- PRIMARY (lead) Delta sdmdc_dom, own alignment ----
    rec, ff = split("dsdmdc")
    primary = mwu(rec, ff, greater=True)
    # co-primary auto_unique
    rec_a, ff_a = split("auto_unique_post")
    coprimary = mwu(rec_a, ff_a, greater=True)
    # robustness: io alignment
    rec_io, ff_io = split("dsdmdc", "io")
    primary_io = mwu(rec_io, ff_io, greater=True)

    # ---- SECONDARIES (Holm over 3) ----
    s1 = mwu(*split("dflow"), greater=True)                 # rec > ff
    s2 = mwu(*split("speed_post"), greater=False)           # rec < ff (settling)
    s3 = mwu(*split("rollout_mse"), greater=False)          # rec < ff
    sec_p = [s1["p"], s2["p"], s3["p"]]
    sec_adj = holm(sec_p)
    secondaries = {
        "dflow_balance(rec>ff)": {**s1, "p_holm": sec_adj[0]},
        "settling_speed(rec<ff)": {**s2, "p_holm": sec_adj[1]},
        "rollout_mse(rec<ff)": {**s3, "p_holm": sec_adj[2]},
    }

    # ---- triviality guard (PREREG sec.9) ----
    pre_disp = [r["own"].get("disp_pre", np.nan) for r in inc]
    pre_disp = [x for x in pre_disp if np.isfinite(x)]
    med_pre = float(np.median(pre_disp)) if pre_disp else np.nan
    eps = 0.05 * med_pre
    frozen = []
    for r in inc:
        dp = r["own"].get("disp_post", np.nan)
        r["frozen"] = bool(np.isfinite(dp) and dp < eps)
        if r["frozen"]:
            frozen.append((r["arch"], r["seed"], r["family"], round(dp, 4)))
    # recompute primary excluding frozen
    nz = [r for r in inc if not r["frozen"]]
    rec_nz = [r["own"].get("dsdmdc", np.nan) for r in nz if r["family"] == "recurrent"]
    ff_nz = [r["own"].get("dsdmdc", np.nan) for r in nz if r["family"] == "feedforward"]
    primary_nonfrozen = mwu(rec_nz, ff_nz, greater=True)

    # architecture-level corroboration (3 rec means vs 3 ff means)
    def arch_means(key):
        out = {}
        for r in inc:
            out.setdefault((r["family"], r["arch"]), []).append(r["own"].get(key, np.nan))
        rec = [np.nanmean(v) for (fam, a), v in out.items() if fam == "recurrent"]
        ff = [np.nanmean(v) for (fam, a), v in out.items() if fam == "feedforward"]
        return rec, ff
    am_rec, am_ff = arch_means("dsdmdc")
    arch_corrob = dict(rec_arch_means=[round(x, 4) for x in am_rec],
                       ff_arch_means=[round(x, 4) for x in am_ff],
                       direction="rec>ff" if np.mean(am_rec) > np.mean(am_ff) else "rec<=ff")

    # ---- VERDICT ----
    p_primary = primary["p"]
    primary_confirms = (p_primary is not None and p_primary < ALPHA)
    survives_trivial = (primary_nonfrozen["p"] is not None and primary_nonfrozen["p"] < ALPHA)
    any_sec = any(a is not None and a < ALPHA for a in sec_adj)
    if primary_confirms and survives_trivial:
        verdict = "(a) PRIMARY CONFIRMS — instrument diagnoses architecture under matched computation"
    elif primary_confirms and not survives_trivial:
        verdict = "(a*) PRIMARY confirms but driven by frozen/degenerate nets — DOWNGRADE to non-diagnostic"
    elif (not primary_confirms) and any_sec:
        verdict = "(b) PRIMARY KILLS, a secondary separates — lead with diagnostic measure, demote dsdmdc"
    else:
        verdict = "(c) NOTHING SEPARATES — implementation interpretation not supported; revise paper"

    out = dict(task=task, alpha=ALPHA, inclusion=incl,
               n_included=len(inc), excluded_by_accuracy=excl,
               family_counts={"recurrent": sum(r["family"] == "recurrent" for r in inc),
                              "feedforward": sum(r["family"] == "feedforward" for r in inc)},
               PRIMARY_dsdmdc_own=primary, COPRIMARY_auto_unique=coprimary,
               PRIMARY_dsdmdc_io_robustness=primary_io,
               SECONDARIES_holm=secondaries,
               triviality=dict(median_pre_disp=round(med_pre, 4), eps=round(eps, 4),
                               frozen_nets=frozen, primary_excluding_frozen=primary_nonfrozen),
               arch_level_corroboration=arch_corrob,
               VERDICT=verdict)
    json.dump(out, open(os.path.join(RES, f"stats_{task}.json"), "w"), indent=1, default=str)
    print(json.dumps({k: out[k] for k in ("task", "n_included", "family_counts",
                                          "PRIMARY_dsdmdc_own", "COPRIMARY_auto_unique",
                                          "SECONDARIES_holm", "VERDICT")}, indent=1, default=str))
    print("\nexcluded_by_accuracy:", excl)
    print("frozen_nets:", frozen)
    print("VERDICT:", verdict)


if __name__ == "__main__":
    main()
