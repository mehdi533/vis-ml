# New research results

Findings produced on-machine while hardening the repo and building follow-up
research modules. All are **mid-scale local** results (small datasets, realistic
but not thesis-final): they establish mechanisms and directions; publication-grade
magnitudes need the cluster dataset. Nothing here is fabricated; every number is
reproducible from the scripts/configs noted.

## 1. Conformal margins close the RoCoF replay gap
The embedded surrogate under-predicts the security metric near the limit ~2/3 of
the time — quantifying the §6.2 replay gap. A split-conformal one-sided margin,
calibrated on real held-out surrogate residuals (250 IEEE-39 sims, α=0.1, 200
splits), lifts *safety coverage* to the 90% target:

| Metric | coverage (raw) | coverage (+ conformal) | margin |
|---|---|---|---|
| RoCoF | 32.1% | 90.0% | 0.135 Hz/s |
| Δf | 33.9% | 89.7% | 0.264 Hz |

Distribution-free, finite-sample. `scripts/run_conformal_demo.py`.
**Novel:** conformal calibration + exact NN embedding for virtual-inertia dispatch.

The miscoverage level α is the operator's dial — tighter α ⇒ higher coverage ⇒
a larger (more conservative) margin:

| α | coverage (after) | RoCoF margin (Hz/s) | Δf margin (Hz) |
|---|---|---|---|
| 0.20 | 0.80 | 0.082 | 0.166 |
| 0.10 | 0.90 | 0.133 | 0.267 |
| 0.05 | 0.95 | 0.251 | 0.444 |

Raw-surrogate coverage stays ~0.32 regardless of α. `apply_conformal_margins.py`
turns a chosen α into a security-tightened optimization config automatically; when
a margin exceeds the envelope half-width it flags that the surrogate is too
inaccurate to certify that bound at the target coverage.

## 2. Embeddability scales favorably
Binaries in the exact ReLU-MILP encoding, over the full input domain vs. the
schedulable (M/D-only) box, on the trained surrogates:

| System | full domain | schedulable box | reduction | big-M |
|---|---|---|---|---|
| IEEE 39 | 288 | 74 | −74% | 8.85 → 1.74 |
| IEEE 118 | 288 | 30 | −90% | 15.4 → 2.03 |

The reduction **grows with system size**. OBBT tightens big-M a further ~31%. At a
fixed operating point (deployment, only M/D free) the embedded net needs ~0
binaries — the MILP is cheap in operation.

## 3. Accuracy ≠ largest embedding (Pareto)
Same data, three architectures. MTLSH is Pareto-dominant:

| Model | agg RMSE ↓ | hidden ReLU ↓ | box binaries ↓ |
|---|---|---|---|
| **MTLSH** | **0.669** | **48** | **1** |
| MLP | 0.685 | 96 | 7 |
| MTLGSH | 0.713 | 96 | 13 |

Quantifies the thesis's headline finding and backs the choice of MTLSH.

## 4. COI hides the worst bus (regional security)
The worst individual bus sees **2.38× the COI RoCoF on average (up to 7.98×)**.
A COI-only constraint can call a schedule safe while a bus violates. A multi-head
surrogate predicts the worst-bus metrics (worst-bus RoCoF norm. RMSE 0.15), so the
dispatch can constrain the regional worst case.

## 5. Deterministic verification
A MILP certifies the surrogate's output range over an entire input box (e.g. RoCoF
provably within a fixed interval across all feasible M/D schedules). Pairs with
conformal: statistical *and* worst-case guarantees. `research/embeddability/verify.py`.

## 6. Scale-up to IEEE 118
The pipeline runs end-to-end on a dynamified IEEE 118 (50 machines + 4 REGCV1
grid-forming IBRs); a low-inertia build (H=2.5) swings realistically (tail RoCoF
0.21 Hz/s, Δf 0.44 Hz). Turns "generalisation asserted" toward "demonstrated".

---

### Remaining (heavier / cluster-gated)
- Wire PTDF/N-1 + a real cost table so the IEEE 118 optimisation is genuinely
  security-constrained (not the reduced formulation).
- Regenerate the full dataset on the cluster → publication-grade magnitudes.
- Decision-focused training (surrogate through the dispatch solution map) and
  Directed-Walks boundary sampling.

See `research/README.md` for module details and the dashboards under
`../visualization/` for the visual summary.
