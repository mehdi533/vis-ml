# Research modules

Follow-up research built on the thesis pipeline (surrogate-assisted virtual
inertia scheduling). Each module is self-contained, numpy/torch/cvxpy-based, and
unit-tested (`pytest tests/`). Results below were produced on-machine on
mid-scale local data (labelled as such); full-scale numbers need the cluster
dataset.

## `research/conformal` — distribution-free robust margins
Closes the RoCoF *replay gap* (thesis §6.2): the embedded surrogate can call a
schedule safe while the true (simulated) response exceeds the limit.

- `calibration.py` — split-conformal one-sided margins (`abs` mode for symmetric
  envelopes like |RoCoF|≤R, `upper` mode otherwise). Finite-sample coverage
  ≥ 1−α, validated by Monte-Carlo tests.
- `apply.py` — adapter from `replay_validation` detail CSVs → per-metric margins
  → tightened `y_min`/`y_max` for re-optimization.

**Result:** on 250 IEEE-39 sims, calibrating on real surrogate residuals lifts
safety coverage from ~32% (raw surrogate under-predicts ~2/3 of the time) to the
90% target for both RoCoF and Δf. Demo: `scripts/run_conformal_demo.py`.

## `research/embeddability` — cheaper, verifiable MILP embeddings
- `bounds.py` — interval bound propagation (IBP) + ReLU-stability accounting
  (how many neurons need a binary, how large big-M must be).
- `obbt.py` — optimization-based bound tightening (LP per neuron over the
  triangle relaxation): tighter than IBP, never wider.
- `verify.py` — MILP worst-case certification of a surrogate output over an
  input box (deterministic companion to the conformal statistical guarantee).

**Results:** restricting to the schedulable box cuts binaries −74% (IEEE 39) /
−90% (IEEE 118) — tightening scales *favorably*. OBBT trims big-M a further ~31%.
At a fixed operating point (deployment), the embedded net needs ~0 binaries.
Scripts: `scripts/analyze_embeddability.py`, `scripts/run_pareto.py`,
`scripts/run_solvetime_experiment.py`.

## `research/headroom` — targeted-vs-uniform allocation analysis
Quantifies the *structural* efficiency story honestly (reallocation, not a
certified saving — the headroom is an inexact proxy per §6.2): headroom freed,
SG→IBR reserve shift, and M/D allocation non-uniformity, from the dispatch-impact
schema.

## `research/systems` — scale beyond IEEE 39-bus
- `registry.py` — declarative `SystemSpec` + registry (ieee39, ieee118, npcc140),
  a `describe_system` diagnostic, and `augment_with_grid_forming_ibrs` (REGCV1).
- `dynamify.py` — assign GENROU + TGOV1N to a power-flow-only case (IEEE 118/300)
  so it can run TDS.

**Result:** the full pipeline (build → data → train → embed → SCIP solve) runs
end-to-end on a dynamified IEEE 118 with 4 grid-forming IBRs; a low-inertia
build (H=2.5) swings realistically. Builder: `scripts/build_ieee118_ibrs.py`.

## Accuracy vs. embeddability (Pareto)
`scripts/run_pareto.py`: MTLSH is Pareto-dominant — best accuracy (agg RMSE
0.669) *and* smallest embedding (48 hidden ReLU, 1 box binary) vs MLP / MTLGSH.

## `research/sampling` — boundary-focused active sampling
`directed_walks.py`: gradient-driven Directed-Walks sampling — drive the
schedulable M/D inputs toward a target output (the security boundary) using the
surrogate's gradient, so generated data lands where the dispatch's active
constraints live. On the trained MTLSH, walks focus samples ~130× closer to the
boundary.

## Safety-oriented training
`models/losses.py` adds `pinball` (quantile) loss: with `tau > 0.5` the surrogate
is penalised more for under-prediction, biasing it to over-state the security
metric — conservative even before a conformal margin.

## Regional security
`scripts/build_regional_dataset.py` + `configs/model/regional_multihead.yaml`:
the worst individual bus sees 2.4× the COI RoCoF (up to 8×); a multi-head
surrogate predicts the worst-bus metrics so the dispatch can constrain the
regional worst case, not just the COI average.
