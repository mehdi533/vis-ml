# ReLU Constraints: MILP (Exact) vs Epigraph (Convex Relaxation)

## Setup
For a ReLU unit with pre-activation \(z\) and output \(y\):
\[
z = W h + b,\quad y = \max(0, z)
\]

This can be embedded in an optimization problem (e.g., ED) as constraints.

---

## Epigraph Reformulation (Convex Relaxation)
Replace the ReLU by its **convex epigraph**:
\[
y \ge 0,\qquad y \ge z
\]

**Properties**
- **Convex**: only linear inequalities.
- **Fast**: solved with QP/SOCP solvers.
- **Relaxation**: does **not** enforce \(y = \max(0,z)\), so \(y\) can be larger than the true ReLU output.
- **Consequence**: NN outputs in the optimization (\(y\)) may **not match** the actual NN prediction for the same input.

---

## MILP Reformulation (Exact ReLU)
Use big‑M with a binary variable \(a \in \{0,1\}\):
\[
y \ge 0,\quad y \ge z
\]
\[
y \le z - z^{\min}(1-a),\quad y \le z^{\max} a
\]

where \(z^{\min}, z^{\max}\) are valid bounds on \(z\).

**Properties**
- **Exact**: enforces \(y = \max(0,z)\).
- **Nonconvex** (MILP): requires MIP solvers (Gurobi/CPLEX/SCIP).
- **Slower**: scales poorly with network size (many binaries).
- **Requires bounds**: must have finite \(z^{\min}, z^{\max}\).

---

## Practical Differences
| Aspect | Epigraph (Convex) | MILP (Exact) |
|---|---|---|
| Correctness | Relaxation (may be loose) | Exact |
| Solver | QP/SOCP (fast) | MIP (slower) |
| Output consistency | \(y\) can differ from NN prediction | \(y\) equals NN prediction |
| Scalability | Good | Poorer |
| Required bounds | No | Yes (big‑M) |

---

## In This Repo
- **Epigraph**: implemented in `scheduling/mtlsh_relu_convex.py`.
- **MILP**: added in `experiments/scan_step_scale_convex.py` (flag `--use-milp`).

---

## Takeaway
Use **epigraph** when you need speed and a convex formulation, but expect relaxed outputs.  
Use **MILP** when you need exact NN behavior and can afford MIP solve time.
