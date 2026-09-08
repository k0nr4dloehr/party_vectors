"""Dose-response test: text divergence from alpha=0 should grow with |alpha|.

Uses normalized token-overlap similarity between steered and alpha=0 responses.
Monotone decrease with alpha => injection scaling works as intended.
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PARTIAL = ROOT / "data" / "results_new" / (sys.argv[1] if len(sys.argv) > 1 else "partial_fix2")
TAG = "meta-llama-3-8b-instruct"
PARTIES = ["CDU_CSU", "GRUENE", "SPD", "AfD", "DIE_LINKE"]


def token_overlap(a: str, b: str) -> float:
    ta, tb = set(str(a).lower().split()), set(str(b).lower().split())
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


rows = []
for party in PARTIES:
    df = pd.read_csv(PARTIAL / f"sweep_ckpt_{TAG}_{party}.csv")
    base = df[df["alpha"] == 0.0].set_index(["layer", "thesis_key"])["steered_response"]
    nz = df[df["alpha"] != 0.0].copy()
    nz["base_text"] = nz.set_index(["layer", "thesis_key"]).index.map(base)
    nz["overlap"] = [
        token_overlap(a, b) for a, b in zip(nz["steered_response"], nz["base_text"])
    ]
    g = nz.groupby("alpha")["overlap"].mean()
    for alpha, mean_overlap in g.items():
        rows.append({"party": party, "alpha": alpha, "mean_overlap": mean_overlap})

res = pd.DataFrame(rows)
piv = res.pivot(index="alpha", columns="party", values="mean_overlap").round(3)
print("Mean token-overlap with alpha=0 response (lower = more change):")
print(piv.to_string())
mono = res.groupby("party").apply(
    lambda d: d.sort_values("alpha")["mean_overlap"].is_monotonic_decreasing,
    include_groups=False,
)
print("\nMonotone decreasing with alpha:")
print(mono.to_string())
