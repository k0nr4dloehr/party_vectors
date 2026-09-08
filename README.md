# Party Vectors: Are Political Parties Represented as Linear Directions in Activation Space?

Code and data accompanying the paper *"Party Vectors: Are Political Parties
Represented as Linear Directions in Activation Space?"*.

This repository contains the pipeline that extracts per-party steering
vectors from the residual streams of eight open-weight instruction-tuned
models, tests them with a judge-free logit probe and judged open-ended
generation, and validates the LLM judge against human annotators. All raw
per-thesis results and the fixed train/selection split are included, so every
number in the paper can be recomputed from the released data.

## Repository layout

```
party_vectors/
├── phase1_baseline.py            Phase 1: party-conditioned generation + similarity weighting
├── phase1_neutral_baseline.py    Neutral-conditioned baseline (provenance)
├── phase2_vectors.py             Phase 2: weighted party-vs-other-parties vector extraction
├── phase3_steering.py            Phase 3: judged open-ended generation sweep
├── phase3b_logit_probe.py        Phase 3b: judge-free logit probe (layer/alpha selection)
├── phase3_rejudge.py             Re-judge pass that fills NaN judge scores
├── utils.py                      Shared helpers (data loading, judge client, hooks)
├── build_split_manifest.py       Provenance for the fixed train/selection split
├── analyze_iaa.py                Inter-annotator agreement analysis
├── sample_human_annotation.py    Annotation sample construction
├── requirements.txt
├── viz/                          Diagnostic and plotting scripts
├── figures/
│   ├── make_paper_figures.py     Single source of truth for all paper figures
│   └── paper/                    The figures as published in the paper (PDF + PNG)
└── data/
    ├── wahl-o-mat-clean.xlsx     Wahl-O-Mat theses (BT21, BT25, EU24), 5 parties
    ├── thesis_split_manifest.csv Fixed train/selection assignment (114 theses)
    ├── vectors_new/              Extracted party vectors per model/layer/party
    ├── results_new/
    │   ├── baseline_results_*.csv    Phase 1 similarity scores
    │   ├── probe/                    Phase 3b logit-probe results (train + selection)
    │   └── steering/                 Phase 3 generation results, summaries, selected configs
    └── human_annotation/         Human-annotation data + gold LLM scores
```

## Data

- **`data/wahl-o-mat-clean.xlsx`** — the Wahl-O-Mat theses for the 2021 and
  2025 federal elections and the 2024 European election, restricted to the
  five factions currently in the German Bundestag (CDU/CSU, SPD, GRÜNE, AfD,
  Die Linke). Source: Bundeszentrale für politische Bildung (bpb).
- **`data/thesis_split_manifest.csv`** — the fixed assignment of all 114
  theses into a train half (57) and a selection half (57). Experiments read
  this file and never redraw the split. The split is balanced by coarse
  policy category and election.
- **`data/vectors_new/`** — one unnormalized steering vector per party per
  layer per model, stored as `vector_layer_XX.pt` with a `metadata.json`
  recording the extraction method and per-statement weights.
- **`data/results_new/probe/`** — per-thesis logit-probe results
  (`margin_cong`, `p3_cong`, `argmax3_is_cong`, `top1_token`) for both
  splits.
- **`data/results_new/steering/`** — per-thesis judged generation results,
  per-(layer, alpha) summaries with 95% bootstrap intervals, and the
  selected configuration per model and party.
- **`data/human_annotation/`** — the human-annotated sample (`gold.csv` with
  LLM scores, `annotators all.xlsx` with human scores) used to validate the
  LLM judge.

## Reproducing the paper's numbers

```bash
# Regenerate every paper figure from the raw results
python figures/make_paper_figures.py
```

The figures referenced by the paper are written to `figures/paper/` and are
included in this repository as both PDF and PNG.

The judge-validation statistics (Section 7 of the paper) are recomputed from
the annotation data by `paper/refresh_judge_stats.py` in the paper's source
repository.

## Running the pipeline

The pipeline phases are chained: Phase 1 produces the similarity weights,
Phase 2 extracts the vectors, Phase 3b selects layers/alphas with the logit
probe, and Phase 3 runs the judged generation sweep.

```bash
# Install dependencies
pip install -r requirements.txt
```

The phases read their configuration from environment variables (see
`utils.py` and the individual phase scripts): `MODEL_KEY`, `DATA_PATH`,
`SPLIT_MANIFEST`, `VECTOR_DIR`, `RESULTS_DIR`, and the judge settings
(`JUDGE_MODEL`, `JUDGE_BASE_URL`, ...). From the repository root, the
defaults resolve as:

```bash
export DATA_PATH=data/wahl-o-mat-clean.xlsx
export SPLIT_MANIFEST=data/thesis_split_manifest.csv
export MODEL_KEY=llama3
```

### Models

`MODEL_REGISTRY` in `utils.py` maps short keys to HuggingFace IDs:
`llama3` (8B, reference model), `llama3-1b`, `llama3-3b`, `llama3-70b`,
`qwen3`, `mistral`, `deepseek`, `gemma`. `MODEL_KEY` selects the model.
`llama3-70b` requires `LOAD_IN_4BIT=1` on a single H100 80GB.

### Method summary

For each party and layer, the party vector is the weighted contrast between
the party-conditioned hidden state and the mean of the other four parties'
states at the last prompt token:

```
v_P = ( Σ_t w_t ( h_P(t) − mean_{Q≠P} h_Q(t) ) ) / Σ_t w_t
```

where `w_t` is the Phase-1 similarity weight of the party-conditioned answer
on thesis `t`. Injection follows Arditi et al. (2024): the unit-normalized
vector is added at every token position, `h ← h + α·v̂`.

## Reproducibility notes

- Every long-running job writes per-unit checkpoints guarded by a SHA-256
  fingerprint of the run parameters; mismatched checkpoints are refused.
- The split assignment is fixed and released; experiments read it and do not
  redraw it.
- The paper's Section 7 (judge validation) is currently based on a pending
  annotation set; the annotation data is included here and the recomputation
  script lives with the paper sources.

## License

- Code: MIT License (see `LICENSE`).
- Data: CC-BY-4.0 (see `LICENSE`). The Wahl-O-Mat theses originate from the
  Bundeszentrale für politische Bildung (bpb); see
  https://www.bpb.de/themen/wahl-o-mat/556865/datensaetze-des-wahl-o-mat/
  for the original dataset terms.

## Citation

If you use this code or data, please cite the paper:

```bibtex
@inproceedings{loehr2026partyvectors,
  title={Party Vectors: Are Political Parties Represented as Linear
         Directions in Activation Space?},
  author={L{\"o}hr, Konrad and Yuan, Shuzhou and F{\"a}rber, Michael},
  year={2026}
}
```