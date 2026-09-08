"""Run all visualization scripts."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from plot_baseline_comparison import main as plot_baseline
from plot_cross_party_profiles import main as plot_profiles
from plot_steering_heatmaps import main as plot_heatmaps
from plot_steering_summary import main as plot_summary
from plot_steering_success import main as plot_success


PLOT_STEPS = [
    ("Steering success (per-model + comparisons)", plot_success),
    ("Steering heatmaps (layer × alpha)", plot_heatmaps),
    ("Baseline vs steering comparison", plot_baseline),
    ("Cross-party alpha profiles", plot_profiles),
    ("Steering summary matrices", plot_summary),
]


def main() -> int:
    failures = []
    for idx, (label, fn) in enumerate(PLOT_STEPS, start=1):
        print("=" * 60)
        print(f"{idx}/{len(PLOT_STEPS)} {label}")
        print("=" * 60)
        try:
            fn()
        except Exception as exc:
            failures.append((label, exc))
            print(f"FAILED: {label}: {exc}")
            traceback.print_exc()
    print("\nAll figures saved to figures/")
    if failures:
        print("\nSome plots failed:")
        for label, exc in failures:
            print(f"  - {label}: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
