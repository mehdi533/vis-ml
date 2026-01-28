# VIS-RTED MILP (Implemented Form)

This repo now includes a concrete VIS-RTED mixed-integer linear program in `scheduling/vis_rted_milp.py`.

It co-optimizes dispatch, reserves, and IBR virtual inertia/damping while enforcing frequency-security surrogates and IBR headroom requirements.

## Decision Variables

At each RTED interval `t`, the MILP decides:

- `p_sg[i]`: synchronous generator dispatch
- `r_up_sg[i]`, `r_down_sg[i]`: SG up/down reserve
- `p_ibr[j]`: IBR dispatch
- `r_ibr[j]`: IBR reserve / headroom for inertial response
- `m_ibr[j]`: IBR virtual inertia
- `d_ibr[j]`: IBR virtual damping

## Core Constraints

The implemented constraints follow the VIS paper structure:

- Power balance: `sum(p_sg) + sum(p_ibr) == load_t`
- SG bounds: `p_min <= p_sg <= p_max`
- SG up-reserve bounds: `0 <= r_up_sg <= ru_max`
- SG down-reserve bounds: `0 <= r_down_sg <= rd_max`
- SG headroom with reserves: `p_sg + r_up_sg <= p_max` and `p_sg - r_down_sg >= p_min`
- IBR bounds: `p_min <= p_ibr <= p_max`
- IBR reserve bounds: `0 <= r_ibr <= reserve_max`
- IBR headroom with reserve: `p_ibr + r_ibr <= p_max` and `p_ibr - r_ibr >= p_min`
- IBR inertia/damping bounds: `m_min <= m_ibr <= m_max` and `d_min <= d_ibr <= d_max`
- Line limits (optional DC shift-factor form): `flows = base_flows + shift_sg @ p_sg + shift_ibr @ p_ibr` with `-line_limits <= flows <= line_limits`

## Frequency Security Constraints

### RoCoF Inertia Floor (Linearized)

RoCoF is enforced via an inertia floor:

- Aggregate inertia: `M_total = base_inertia + inertia_weights @ m_ibr`
- RoCoF inertia floor: `rocof_limit * M_total >= f0 * abs(delta_p_e)`

This is linear because `delta_p_e` is treated as known at each interval.

### Nadir and Peak Power via DNN Surrogates

Two ReLU networks are embedded as MILP constraints:

- Peak inertial power surrogate `peak_surrogate`
- Frequency nadir surrogate `nadir_surrogate`

Their inputs are affine functions of decision variables:

- `x = bias + W_sg p_sg + W_ibr p_ibr + W_r ... + W_m m_ibr + W_d d_ibr`

The default feature map is:

- `x = [delta_p_e, M_total, D_total]`

Outputs are enforced as:

- IBR headroom: `r_ibr >= peak_pred` (or equality band if enabled)
- Nadir limit: `-delta_f_limit <= nadir_pred <= delta_f_limit`

## ReLU-to-MILP Encoding

Each hidden ReLU neuron is encoded with a standard big-M formulation using a binary variable `delta`.

For pre-activation `z` with bounds `l <= z <= u` and activation `y = max(0, z)`:

- `y >= z`
- `y >= 0`
- `y <= z - l * (1 - delta)`
- `y <= u * delta`

Interval bound propagation is used to compute per-neuron `(l, u)`.

## Objective

The objective minimizes energy and reserve costs with large penalties on security slack:

- SG energy and reserve costs
- IBR energy and reserve costs
- Slack penalties for RoCoF, nadir, and peak headroom constraints

## How To Use It

Main entry point:

- `scheduling/vis_rted_milp.py:solve_vis_rted_milp`

Key inputs:

- `GeneratorSpec`, `IBRSpec`
- `FrequencySecurityLimits`
- `ReLUNetwork` surrogates
- Optional `NetworkSpec`
- Optional `AffineFeatureMap`

Minimal sketch:

```python
import numpy as np
from scheduling.vis_rted_milp import (
    GeneratorSpec,
    IBRSpec,
    FrequencySecurityLimits,
    ReLUNetwork,
    solve_vis_rted_milp,
)

# Build specs, surrogates, and limits...
result = solve_vis_rted_milp(
    load_t=1.0,
    delta_p_e=0.05,
    generators=[...],
    ibrs=[...],
    freq_limits=FrequencySecurityLimits(f0=50.0, rocof_limit=0.5, delta_f_limit=0.8),
    peak_surrogate=ReLUNetwork(weights=[...], biases=[...]),
    nadir_surrogate=ReLUNetwork(weights=[...], biases=[...]),
)
```

## Important Environment Note

This MILP requires `cvxpy` and a MILP-capable solver. The implementation defaults to `GUROBI`.
