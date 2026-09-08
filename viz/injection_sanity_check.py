"""Injection sanity test: do steered response texts differ from alpha=0 at all?

With greedy decoding, a working hook can still yield identical text if the
perturbation never flips an argmax. But across ~57 theses x 6 alphas x 13
layers, ZERO text changes everywhere would strongly suggest broken injection.
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PARTIAL = ROOT / "data" / "results_new" / "partial_fix2"
TAG = "meta-llama-3-8b-instruct"
PARTIES = ["CDU_CSU", "GRUENE", "SPD", "AfD", "DIE_LINKE"]

for party in PARTIES:
    df = pd.read_csv(PARTIAL / f"sweep_ckpt_{TAG}_{party}.csv")
    text_col = next((c for c in df.columns if "response" in c.lower()), None)
    if text_col is None:
        print(f"{party}: NO response column; columns={list(df.columns)[:12]}")
        continue
    base = df[df["alpha"] == 0.0].set_index(["layer", "thesis_key"])
    nz = df[df["alpha"] != 0.0].copy()
    merged = nz.join(
        base[[text_col]].rename(columns={text_col: "base_text"}),
        on=["layer", "thesis_key"],
    )
    merged["identical"] = merged[text_col].fillna("") == merged["base_text"].fillna("")
    by_layer = merged.groupby("layer")["identical"].agg(["mean", "count"])
    overall = merged["identical"].mean()
    # score movement
    sim_cols = [c for c in df.columns if "sim" in c.lower() or "score" in c.lower()]
    print(f"\n=== {party} (text col: {text_col}; sim cols: {sim_cols[:3]}) ===")
    print(f"  overall identical-text fraction (alpha!=0 vs alpha=0): {overall:.3f}")
    for layer, row in by_layer.iterrows():
        print(f"  L{int(layer):02d}: identical={row['mean']:.2f} (n={int(row['count'])})")
