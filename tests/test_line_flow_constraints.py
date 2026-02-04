import sys
import types
from pathlib import Path
import importlib.util

import numpy as np
import cvxpy as cp

# Stub andes to avoid hard dependency during unit tests
sys.modules.setdefault("andes", types.SimpleNamespace(System=object))

_MOD_PATH = Path(__file__).resolve().parents[1] / "scheduling" / "line_flow_constraints.py"
spec = importlib.util.spec_from_file_location("line_flow_constraints", _MOD_PATH)
mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules["line_flow_constraints"] = mod
spec.loader.exec_module(mod)


def test_build_injection_matrices_basic():
    class _Bus:
        def __init__(self, v):
            self.v = v

    class _Group:
        def __init__(self, buses):
            self.bus = _Bus(buses)

    class Dummy:
        PV = _Group([1])
        Slack = _Group([3])
        PQ = _Group([2, 3])

    bus_ids = [1, 2, 3]
    Cg, Cd = mod.build_injection_matrices(Dummy(), bus_ids=bus_ids)

    assert Cg.shape == (3, 2)
    assert Cd.shape == (3, 2)

    # PV at bus 1, Slack at bus 3
    assert np.allclose(Cg, np.array([[1, 0], [0, 0], [0, 1]]))
    # Loads at bus 2 and 3
    assert np.allclose(Cd, np.array([[0, 0], [1, 0], [0, 1]]))


def test_compute_net_injections_and_flows():
    Cg = np.array([[1.0, 0.0], [0.0, 1.0]])
    Cd = np.array([[1.0, 0.0], [0.0, 1.0]])
    Pg = np.array([2.0, 3.0])
    Pd = np.array([1.5, 0.5])

    p = mod.compute_net_injections(Cg, Pg, Cd, Pd)
    assert np.allclose(p, np.array([0.5, 2.5]))

    ptdf = np.array([[1.0, -1.0]])
    f = mod.compute_line_flows(ptdf, p)
    assert np.allclose(f, np.array([-2.0]))


def test_build_line_flow_constraints_symmetric():
    flows = cp.Variable(2)
    cons = mod.build_line_flow_constraints(flows, fmax=[5.0, 6.0])
    assert len(cons) == 2


def test_build_line_flow_constraints_asymmetric():
    flows = cp.Variable(2)
    cons = mod.build_line_flow_constraints(flows, fmax=[5.0, 6.0], fmin=[-1.0, -2.0])
    assert len(cons) == 2


def test_build_power_balance_constraint():
    inj = cp.Variable(3)
    cons_sys = mod.build_power_balance_constraint(inj, per_bus=False)
    cons_bus = mod.build_power_balance_constraint(inj, per_bus=True)
    assert len(cons_sys) == 1
    assert len(cons_bus) == 1
