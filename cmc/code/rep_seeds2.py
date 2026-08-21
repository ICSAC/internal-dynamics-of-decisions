#!/usr/bin/env python3
"""Pre-authorized fresh-seed replication driver (PREREG-CMC.md sec.10,11: SEEDS2=[6..11]).

This is NOT part of the frozen-hashed analysis code: it only ORCHESTRATES the frozen train.train_one
with the pre-registered confirmatory seeds and writes a SEEDS2 artifacts set. It computes no dynamical
measure and changes no metric/window/decision rule. The frozen analyze_cmc.py + stats_cmc.py are then
run unchanged against this set. Usage: python rep_seeds2.py <task>
"""
import os, sys, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import train, task as TASK
from models import ARCHS

SEEDS2 = [6, 7, 8, 9, 10, 11]
OUT = os.environ.get("CMC_OUT", "./artifacts_rep")


def main():
    t = sys.argv[1] if len(sys.argv) > 1 else "primary"
    os.makedirs(OUT, exist_ok=True)
    data = train.get_data(t)
    (_, _), (_, _), (Ute_s, yte, gte, Ute_raw), _ = data
    tc_io = TASK.ideal_observer_commit(Ute_raw, t)
    np.savez_compressed(os.path.join(OUT, f"shared_{t}.npz"),
                        U=Ute_raw.astype(np.float64), y=yte, gamma=gte, tc_io=tc_io)
    table = []
    for arch in ARCHS:
        for seed in SEEDS2:
            t0 = time.time()
            r = train.train_one(t, arch, seed, data)
            np.savez_compressed(os.path.join(OUT, f"net_{t}_{arch}_{seed}.npz"), X=r["X"], P=r["P"])
            row = {k: r[k] for k in ("arch", "family", "seed", "task", "width", "nparams",
                                     "test_acc", "val_acc", "hidden")}
            table.append(row)
            print(f"REP {t} {arch:4s} s{seed} acc={r['test_acc']:.4f} ({time.time()-t0:.0f}s)", flush=True)
    json.dump(table, open(os.path.join(OUT, f"acc_table_{t}.json"), "w"), indent=1)
    accs = [r["test_acc"] for r in table]
    print(f"REP {t}: acc range [{min(accs):.4f},{max(accs):.4f}] mean={np.mean(accs):.4f}", flush=True)


if __name__ == "__main__":
    main()
