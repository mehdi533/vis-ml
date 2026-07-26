# New research results

Findings produced on-machine while hardening the repo and building follow-up
research modules. All are **mid-scale local** results (small datasets, realistic
but not thesis-final): they establish mechanisms and directions; publication-grade
magnitudes need the cluster dataset. Nothing here is fabricated; every number is
reproducible from the scripts/configs noted.

## 1. Conformal margins close the RoCoF replay gap
The embedded surrogate under-predicts the security metric near the limit roughly
half the time (more for a weaker surrogate) — quantifying the §6.2 replay gap. A
split-conformal one-sided margin, calibrated on real held-out surrogate residuals
(400 IEEE-39 sims, n=60 held-out, α=0.1, 200 splits), lifts *safety coverage* to
the 90% target:

| Metric | coverage (raw) | coverage (+ conformal) | margin |
|---|---|---|---|
| RoCoF | 49% | 90% | 0.047 Hz/s |
| Δf | 52% | 90% | 0.131 Hz |

Distribution-free, finite-sample. `scripts/run_conformal_demo.py`.
**Novel:** conformal calibration + exact NN embedding for virtual-inertia dispatch.

The miscoverage level α is the operator's dial — tighter α ⇒ higher coverage ⇒
a larger (more conservative) margin:

| α | coverage (after) | RoCoF margin (Hz/s) | Δf margin (Hz) |
|---|---|---|---|
| 0.20 | 0.80 | 0.032 | 0.078 |
| 0.10 | 0.90 | 0.047 | 0.131 |
| 0.05 | 0.97 | 0.080 | 0.157 |

Raw-surrogate coverage stays ~0.50 regardless of α; the margin **shrinks as the
surrogate improves** (RoCoF 0.135 → 0.047 Hz/s going from 250 → 400 training
sims) while coverage always meets the 1−α target — the correction is only as
large as the surrogate needs. `apply_conformal_margins.py`
turns a chosen α into a security-tightened optimization config automatically; when
a margin exceeds the envelope half-width it flags that the surrogate is too
inaccurate to certify that bound at the target coverage.

**Robustness (3 surrogate seeds):** coverage-after = 0.90 ± 0.00 (the guarantee is
seed-invariant) and the margin is stable (RoCoF 0.044 ± 0.005 Hz/s, Δf
0.119 ± 0.010 Hz), while *raw* coverage swings 0.37–0.61 across seeds — the raw
surrogate's safety is luck-of-the-draw; the conformal margin makes it reliable.

**All six embedded outputs:** the lift holds uniformly — raw coverage 0.31–0.51 →
90% target for every metric. The converter-power margins are larger (~0.76–0.95
p.u.), reflecting the surrogate's harder ΔP channels, so those bounds cost more
headroom to certify — an honest, actionable signal.

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
| **MTLSH** (ReLU) | **0.669** | 48 | 1 |
| MLP (ReLU) | 0.685 | 96 | 7 |
| MTLGSH (ReLU) | 0.713 | 96 | 13 |
| FICNN (convex) | 6.43 | 0 | 0 |
| PICNN (convex) | 13.88 | 0 | 0 |

The convex families embed for free (0 binaries) but are **10–20× less accurate**
on this non-convex response surface — far off the useful part of the front. The
ReLU multitask families occupy the useful region, and MTLSH is Pareto-dominant
within it. This is the quantitative form of the thesis's "most accurate ≠ largest
embedding" finding and its choice of MTLSH. (Convex accuracy may improve with more
capacity, but the structural gap is large and consistent with the thesis.)

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
