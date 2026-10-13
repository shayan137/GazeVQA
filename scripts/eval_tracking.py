#!/usr/bin/env python3
"""
compare gaze tracking runs (e.g. dense baseline vs the tracking agent):
AUC, average L2, in/out AP and the number of gaze estimator calls
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd

from source.tracking.metrics import summarize


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("predictions", nargs="+", help="prediction CSVs from scripts/run_tracking.py")
    args = ap.parse_args()

    print(f"{'run':<40}{'calls':>8}{'frames':>8}{'AUC':>9}{'Avg L2':>9}{'Inout AP':>10}")
    for path in args.predictions:
        s = summarize(pd.read_csv(path, dtype={"person_id": str}))
        name = os.path.splitext(os.path.basename(path))[0]
        print(f"{name:<40}{s['estimator_calls']:>8}{s['frames']:>8}{s['auc']:>9.4f}{s['l2']:>9.4f}{s['inout_ap']:>10.4f}")


if __name__ == "__main__":
    main()
