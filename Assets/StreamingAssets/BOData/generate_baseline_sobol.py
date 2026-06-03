# generate_baseline_sobol.py
# ---------------------------------------------------------------------------
# Generates the deterministic baseline design matrix used by the three
# baseline scale blocks (Likert 5 / 20 / 100) in the Fitts-law study.
#
# The points are drawn in the unit hypercube [0, 1]^d with botorch's
# draw_sobol_samples -- the *same* low-discrepancy Sobol sequence that bo.py
# uses for its initial-sampling phase (see bo.py: draw_sobol_samples(...)).
#
# Why the unit cube (and not the raw parameter ranges)?
#   The Sobol points are bounds-independent. Unity (FittsLawTask) denormalizes
#   each column to the *live* BoForUnityManager parameter bounds at apply time.
#   Because all three baseline blocks replay the SAME unit-cube matrix against
#   the SAME bounds, they produce byte-identical designs -- 10 designs total,
#   not 30. Indexing is by round number, so the sequence cannot drift.
#
# Output: baseline_sobol_unit.csv
#   - ';'-separated, one column per design parameter (header == parameter key)
#   - N rows of values in [0, 1]
#   - column order / keys MUST match the BoForUnityManager parameter list.
#     If you change the study's parameters (count, keys, or order), re-run this
#     script with matching --columns so the Sobol dimension stays aligned.
#
# Usage:
#   python generate_baseline_sobol.py                 # 10 rows, seed 3, default cols
#   python generate_baseline_sobol.py --n 10 --seed 3
# ---------------------------------------------------------------------------

import argparse
import csv
import os

import torch
from botorch.utils.sampling import draw_sobol_samples

# Must match BoForUnityManager.parameters (FittsLawTask design parameter keys),
# in the same order, so Sobol dimension j maps to parameter j.
DEFAULT_COLUMNS = [
    "x_font_size",
    "button_size",
    "button_distance",
    "button_hue",
    "button_saturation",
]


def main():
    ap = argparse.ArgumentParser(description="Generate the baseline Sobol design matrix (unit cube).")
    ap.add_argument("--n", type=int, default=10, help="Number of baseline designs (rounds per scale).")
    ap.add_argument("--seed", type=int, default=3, help="Sobol seed (matches bo.py SEED default).")
    ap.add_argument("--columns", nargs="+", default=DEFAULT_COLUMNS,
                    help="Design-parameter keys, in BoForUnityManager order.")
    ap.add_argument("--out", default=None, help="Output CSV path (defaults next to this script).")
    args = ap.parse_args()

    d = len(args.columns)
    bounds = torch.stack([
        torch.zeros(d, dtype=torch.double),
        torch.ones(d, dtype=torch.double),
    ])
    # n=1 batch, q=args.n quasi-random points -> shape (1, n, d) -> (n, d).
    points = draw_sobol_samples(bounds=bounds, n=1, q=args.n, seed=args.seed).squeeze(0)

    out_path = args.out or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "baseline_sobol_unit.csv"
    )
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(args.columns)
        for row in points.tolist():
            writer.writerow([f"{v:.6f}" for v in row])

    print(f"Wrote {args.n}x{d} baseline Sobol design (seed={args.seed}) to:\n  {out_path}")


if __name__ == "__main__":
    main()
