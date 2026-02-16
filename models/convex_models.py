# convex_models.py
# Three convex-in-input architectures for optimization embedding:
#   1) FICNN: convex in all inputs x
#   2) PICNN: convex in u, unconstrained in v (context)
#   3) PICNN-MTLSH: PICNN "shared trunk + per-task heads" (convex in u for each task)
#
# Notes:
# - Convexity relies on ReLU (convex + nondecreasing) and Wz >= 0 on paths that mix convex
#   hidden features z. We enforce Wz >= 0 via softplus reparam (NonNegLinear).
# - We do NOT use dropout in these models (dropout breaks deterministic convexity).
# - Multi-output is "componentwise convex": each output coordinate is convex in u (or x).

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Utilities / constrained layers
# -----------------------------

class NonNegLinear(nn.Linear):
    """Linear layer with non-negative effective weights via softplus reparam."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_pos = F.softplus(self.weight)  # elementwise >= 0
        return F.linear(x, w_pos, self.bias)


def _act(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "softplus":
        return nn.Softplus()
    raise ValueError(f"Unsupported activation: {name}")


# -----------------------------
# 1) FICNN (Fully Input Convex)
# -----------------------------

class FICNN(nn.Module):
    """
    Fully Input Convex Neural Network (Amos & Kolter style).

    Convex in x.
    Structure:
        z0 = g(Wx0 x + b0)
        z_{k+1} = g(Wz_k z_k + Wx_k x + b_k)   with Wz_k >= 0
        y = Wz_out z_L + Wx_out x + b_out      with Wz_out >= 0

    Output can be multi-dim => componentwise convex.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_sizes: Sequence[int] = (256, 128, 64),
        activation: str = "relu",
    ):
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.hidden_sizes = list(hidden_sizes)
        self.g = _act(activation)

        if len(self.hidden_sizes) == 0:
            # Pure affine map is convex
            self.out_wx = nn.Linear(self.in_dim, self.out_dim)
            self.first_wx = None
            self.Wz_layers = nn.ModuleList()
            self.Wx_layers = nn.ModuleList()
            self.out_wz = None
            return

        self.first_wx = nn.Linear(self.in_dim, self.hidden_sizes[0])

        self.Wz_layers = nn.ModuleList()
        self.Wx_layers = nn.ModuleList()
        for i in range(1, len(self.hidden_sizes)):
            self.Wz_layers.append(NonNegLinear(self.hidden_sizes[i - 1], self.hidden_sizes[i]))
            self.Wx_layers.append(nn.Linear(self.in_dim, self.hidden_sizes[i]))

        self.out_wz = NonNegLinear(self.hidden_sizes[-1], self.out_dim)
        self.out_wx = nn.Linear(self.in_dim, self.out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.first_wx is None:
            return self.out_wx(x)

        z = self.g(self.first_wx(x))
        for Wz, Wx in zip(self.Wz_layers, self.Wx_layers):
            z = self.g(Wz(z) + Wx(x))

        return self.out_wz(z) + self.out_wx(x)


# -----------------------------
# 2) PICNN (Partially Input Convex)
# -----------------------------

class PICNN(nn.Module):
    """
    Partially Input Convex Neural Network.

    Split input into:
        u: decision variables (convex w.r.t. u)
        v: context/parameters (no convexity guarantee in v)

    Convex in u for any fixed v.
    Structure:
        z0 = g(Wu0 u + Wv0 v + b0)                 (affine in u, v)
        z_{k+1} = g(Wz_k z_k + Wu_k u + Wv_k v + b_k)  with Wz_k >= 0
        y = Wz_out z_L + Wu_out u + Wv_out v + b_out   with Wz_out >= 0

    Output can be multi-dim => each output coordinate is convex in u.
    """

    def __init__(
        self,
        u_dim: int,
        v_dim: int,
        out_dim: int,
        hidden_sizes: Sequence[int] = (256, 128, 64),
        activation: str = "relu",
    ):
        super().__init__()
        self.u_dim = int(u_dim)
        self.v_dim = int(v_dim)
        self.out_dim = int(out_dim)
        self.hidden_sizes = list(hidden_sizes)
        self.g = _act(activation)

        if len(self.hidden_sizes) == 0:
            # Pure affine in (u,v); convex in u
            self.out_wu = nn.Linear(self.u_dim, self.out_dim)
            self.out_wv = nn.Linear(self.v_dim, self.out_dim) if self.v_dim > 0 else None
            self.first_wu = None
            self.first_wv = None
            self.Wz_layers = nn.ModuleList()
            self.Wu_layers = nn.ModuleList()
            self.Wv_layers = nn.ModuleList()
            self.out_wz = None
            return

        self.first_wu = nn.Linear(self.u_dim, self.hidden_sizes[0])
        self.first_wv = nn.Linear(self.v_dim, self.hidden_sizes[0]) if self.v_dim > 0 else None

        self.Wz_layers = nn.ModuleList()
        self.Wu_layers = nn.ModuleList()
        self.Wv_layers = nn.ModuleList()
        for i in range(1, len(self.hidden_sizes)):
            self.Wz_layers.append(NonNegLinear(self.hidden_sizes[i - 1], self.hidden_sizes[i]))
            self.Wu_layers.append(nn.Linear(self.u_dim, self.hidden_sizes[i]))
            self.Wv_layers.append(nn.Linear(self.v_dim, self.hidden_sizes[i]) if self.v_dim > 0 else None)

        self.out_wz = NonNegLinear(self.hidden_sizes[-1], self.out_dim)
        self.out_wu = nn.Linear(self.u_dim, self.out_dim)
        self.out_wv = nn.Linear(self.v_dim, self.out_dim) if self.v_dim > 0 else None

    def forward(self, u: torch.Tensor, v: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.v_dim > 0:
            if v is None:
                raise ValueError("PICNN expects v when v_dim > 0.")
            if v.shape[-1] != self.v_dim:
                raise ValueError(f"v last dim {v.shape[-1]} != v_dim {self.v_dim}")
        else:
            # allow v=None, ignore
            v = None

        if self.first_wu is None:
            out = self.out_wu(u)
            if self.out_wv is not None:
                out = out + self.out_wv(v)  # type: ignore[arg-type]
            return out

        z = self.first_wu(u)
        if self.first_wv is not None:
            z = z + self.first_wv(v)  # type: ignore[arg-type]
        z = self.g(z)

        for Wz, Wu, Wv in zip(self.Wz_layers, self.Wu_layers, self.Wv_layers):
            h = Wz(z) + Wu(u)
            if Wv is not None:
                h = h + Wv(v)  # type: ignore[arg-type]
            z = self.g(h)

        out = self.out_wz(z) + self.out_wu(u)
        if self.out_wv is not None:
            out = out + self.out_wv(v)  # type: ignore[arg-type]
        return out


# -----------------------------------------
# 3) PICNN with MTLSH spirit (shared + heads)
# -----------------------------------------

@dataclass(frozen=True)
class PICNNGroupedSpec:
    """Optional extra shared blocks applied to a subset of heads."""
    head_indices: Sequence[int]
    hidden_sizes: Sequence[int]


class PICNN_MTLSH(nn.Module):
    """
    PICNN with "shared trunk + per-task heads" (MTL shared-heads spirit).

    Inputs: u (convex), v (context)
    Output: [B, n_tasks] with each task output convex in u (for fixed v).

    Design:
      - A shared PICNN trunk produces a convex hidden feature z_shared(u,v)
      - Each head is a small PICNN block starting from z_shared (convex), mixing u,v via affine skips
      - All "z -> z" weights are enforced nonnegative (NonNegLinear)

    This is solver-friendly: each head output is convex in u.
    """

    def __init__(
        self,
        u_dim: int,
        v_dim: int,
        n_tasks: int,
        trunk_sizes: Sequence[int] = (256, 128),
        head_sizes: Sequence[int] = (128, 64),
        activation: str = "relu",
        group_shared: Optional[Sequence[PICNNGroupedSpec]] = None,
    ):
        super().__init__()
        self.u_dim = int(u_dim)
        self.v_dim = int(v_dim)
        self.n_tasks = int(n_tasks)
        self.g = _act(activation)

        # ---- Shared trunk (PICNN) producing z_shared
        self.trunk = _PICNNTrunk(
            u_dim=self.u_dim,
            v_dim=self.v_dim,
            hidden_sizes=trunk_sizes,
            activation=activation,
        )
        base_dim = self.trunk.out_dim

        # ---- Optional grouped extra shared blocks (still PICNN-safe in u)
        self.group_blocks = nn.ModuleList()
        self.group_block_indices: List[Tuple[int, ...]] = []
        head_input_dims = [base_dim] * self.n_tasks

        if group_shared:
            for spec in group_shared:
                if not spec.head_indices or not spec.hidden_sizes:
                    continue
                indices = tuple(int(i) for i in spec.head_indices)
                for i in indices:
                    if i < 0 or i >= self.n_tasks:
                        raise IndexError(f"Head index {i} out of range [0,{self.n_tasks}).")

                # all these heads must have same current input dim
                d0 = head_input_dims[indices[0]]
                if any(head_input_dims[i] != d0 for i in indices):
                    raise ValueError("Grouped heads must share the same current dim.")

                block = _PICNNFromZ(
                    z_in_dim=d0,
                    u_dim=self.u_dim,
                    v_dim=self.v_dim,
                    hidden_sizes=spec.hidden_sizes,
                    activation=activation,
                )
                self.group_blocks.append(block)
                self.group_block_indices.append(indices)

                for i in indices:
                    head_input_dims[i] = block.out_dim

        # ---- Per-task heads (PICNN blocks from z)
        self.heads = nn.ModuleList([
            _PICNNHead(
                z_in_dim=head_input_dims[i],
                u_dim=self.u_dim,
                v_dim=self.v_dim,
                hidden_sizes=head_sizes,
                activation=activation,
            )
            for i in range(self.n_tasks)
        ])

    def forward(self, u: torch.Tensor, v: Optional[torch.Tensor] = None) -> torch.Tensor:
        z = self.trunk(u, v)  # [B, base_dim]

        head_zs = [z] * self.n_tasks
        for block, indices in zip(self.group_blocks, self.group_block_indices):
            for i in indices:
                head_zs[i] = block(head_zs[i], u, v)

        outs = [self.heads[i](head_zs[i], u, v) for i in range(self.n_tasks)]
        return torch.cat(outs, dim=1)


# -----------------------------
# Internal PICNN building blocks
# -----------------------------

class _PICNNTrunk(nn.Module):
    """PICNN trunk that outputs a convex feature z (vector) in u."""
    def __init__(
        self,
        u_dim: int,
        v_dim: int,
        hidden_sizes: Sequence[int],
        activation: str = "relu",
    ):
        super().__init__()
        self.u_dim = int(u_dim)
        self.v_dim = int(v_dim)
        self.hidden_sizes = list(hidden_sizes)
        self.g = _act(activation)

        if len(self.hidden_sizes) == 0:
            # Degenerate trunk: no hidden state, return zeros? Better: affine map then activation off.
            # We'll just do a single affine -> (no activation) as "z".
            self.out_dim = 0
            self.first_wu = None
            self.first_wv = None
            self.Wz_layers = nn.ModuleList()
            self.Wu_layers = nn.ModuleList()
            self.Wv_layers = nn.ModuleList()
            return

        self.first_wu = nn.Linear(self.u_dim, self.hidden_sizes[0])
        self.first_wv = nn.Linear(self.v_dim, self.hidden_sizes[0]) if self.v_dim > 0 else None

        self.Wz_layers = nn.ModuleList()
        self.Wu_layers = nn.ModuleList()
        self.Wv_layers = nn.ModuleList()
        for i in range(1, len(self.hidden_sizes)):
            self.Wz_layers.append(NonNegLinear(self.hidden_sizes[i - 1], self.hidden_sizes[i]))
            self.Wu_layers.append(nn.Linear(self.u_dim, self.hidden_sizes[i]))
            self.Wv_layers.append(nn.Linear(self.v_dim, self.hidden_sizes[i]) if self.v_dim > 0 else None)

        self.out_dim = self.hidden_sizes[-1]

    def forward(self, u: torch.Tensor, v: Optional[torch.Tensor]) -> torch.Tensor:
        if self.v_dim > 0:
            if v is None:
                raise ValueError("PICNN trunk expects v when v_dim > 0.")
        else:
            v = None

        z = self.first_wu(u)  # type: ignore[operator]
        if self.first_wv is not None:
            z = z + self.first_wv(v)  # type: ignore[arg-type]
        z = self.g(z)

        for Wz, Wu, Wv in zip(self.Wz_layers, self.Wu_layers, self.Wv_layers):
            h = Wz(z) + Wu(u)
            if Wv is not None:
                h = h + Wv(v)  # type: ignore[arg-type]
            z = self.g(h)

        return z


class _PICNNFromZ(nn.Module):
    """Extra shared PICNN-safe block applied to z (convex) with affine u,v skips."""
    def __init__(
        self,
        z_in_dim: int,
        u_dim: int,
        v_dim: int,
        hidden_sizes: Sequence[int],
        activation: str = "relu",
    ):
        super().__init__()
        self.z_in_dim = int(z_in_dim)
        self.u_dim = int(u_dim)
        self.v_dim = int(v_dim)
        self.hidden_sizes = list(hidden_sizes)
        self.g = _act(activation)

        if len(self.hidden_sizes) == 0:
            self.out_dim = self.z_in_dim
            self.first_wz = None
            self.first_wu = None
            self.first_wv = None
            self.Wz_layers = nn.ModuleList()
            self.Wu_layers = nn.ModuleList()
            self.Wv_layers = nn.ModuleList()
            return

        self.first_wz = NonNegLinear(self.z_in_dim, self.hidden_sizes[0])
        self.first_wu = nn.Linear(self.u_dim, self.hidden_sizes[0])
        self.first_wv = nn.Linear(self.v_dim, self.hidden_sizes[0]) if self.v_dim > 0 else None

        self.Wz_layers = nn.ModuleList()
        self.Wu_layers = nn.ModuleList()
        self.Wv_layers = nn.ModuleList()
        for i in range(1, len(self.hidden_sizes)):
            self.Wz_layers.append(NonNegLinear(self.hidden_sizes[i - 1], self.hidden_sizes[i]))
            self.Wu_layers.append(nn.Linear(self.u_dim, self.hidden_sizes[i]))
            self.Wv_layers.append(nn.Linear(self.v_dim, self.hidden_sizes[i]) if self.v_dim > 0 else None)

        self.out_dim = self.hidden_sizes[-1]

    def forward(self, z_in: torch.Tensor, u: torch.Tensor, v: Optional[torch.Tensor]) -> torch.Tensor:
        if self.v_dim > 0 and v is None:
            raise ValueError("PICNNFromZ expects v when v_dim > 0.")

        if self.first_wz is None:
            return z_in

        z = self.first_wz(z_in) + self.first_wu(u)
        if self.first_wv is not None:
            z = z + self.first_wv(v)  # type: ignore[arg-type]
        z = self.g(z)

        for Wz, Wu, Wv in zip(self.Wz_layers, self.Wu_layers, self.Wv_layers):
            h = Wz(z) + Wu(u)
            if Wv is not None:
                h = h + Wv(v)  # type: ignore[arg-type]
            z = self.g(h)
        return z


class _PICNNHead(nn.Module):
    """Per-task head: PICNN-safe mapping from (z,u,v) to scalar output (convex in u)."""
    def __init__(
        self,
        z_in_dim: int,
        u_dim: int,
        v_dim: int,
        hidden_sizes: Sequence[int],
        activation: str = "relu",
    ):
        super().__init__()
        self.z_in_dim = int(z_in_dim)
        self.u_dim = int(u_dim)
        self.v_dim = int(v_dim)
        self.hidden_sizes = list(hidden_sizes)
        self.g = _act(activation)

        # If no head hidden layers: output is affine in u,v plus nonneg linear in z
        self.out_wz = NonNegLinear(self.z_in_dim, 1)
        self.out_wu = nn.Linear(self.u_dim, 1)
        self.out_wv = nn.Linear(self.v_dim, 1) if self.v_dim > 0 else None

        if len(self.hidden_sizes) == 0:
            self.first_wz = None
            self.first_wu = None
            self.first_wv = None
            self.Wz_layers = nn.ModuleList()
            self.Wu_layers = nn.ModuleList()
            self.Wv_layers = nn.ModuleList()
            self.mid_out_wz = None
            return

        self.first_wz = NonNegLinear(self.z_in_dim, self.hidden_sizes[0])
        self.first_wu = nn.Linear(self.u_dim, self.hidden_sizes[0])
        self.first_wv = nn.Linear(self.v_dim, self.hidden_sizes[0]) if self.v_dim > 0 else None

        self.Wz_layers = nn.ModuleList()
        self.Wu_layers = nn.ModuleList()
        self.Wv_layers = nn.ModuleList()
        for i in range(1, len(self.hidden_sizes)):
            self.Wz_layers.append(NonNegLinear(self.hidden_sizes[i - 1], self.hidden_sizes[i]))
            self.Wu_layers.append(nn.Linear(self.u_dim, self.hidden_sizes[i]))
            self.Wv_layers.append(nn.Linear(self.v_dim, self.hidden_sizes[i]) if self.v_dim > 0 else None)

        # map last hidden to output via nonneg in z-space
        self.mid_out_wz = NonNegLinear(self.hidden_sizes[-1], 1)

    def forward(self, z_in: torch.Tensor, u: torch.Tensor, v: Optional[torch.Tensor]) -> torch.Tensor:
        if self.v_dim > 0 and v is None:
            raise ValueError("PICNNHead expects v when v_dim > 0.")

        if self.first_wz is None:
            out = self.out_wz(z_in) + self.out_wu(u)
            if self.out_wv is not None:
                out = out + self.out_wv(v)  # type: ignore[arg-type]
            return out

        z = self.first_wz(z_in) + self.first_wu(u)
        if self.first_wv is not None:
            z = z + self.first_wv(v)  # type: ignore[arg-type]
        z = self.g(z)

        for Wz, Wu, Wv in zip(self.Wz_layers, self.Wu_layers, self.Wv_layers):
            h = Wz(z) + Wu(u)
            if Wv is not None:
                h = h + Wv(v)  # type: ignore[arg-type]
            z = self.g(h)

        out = self.mid_out_wz(z) + self.out_wz(z_in) + self.out_wu(u)
        if self.out_wv is not None:
            out = out + self.out_wv(v)  # type: ignore[arg-type]
        return out


# -----------------------------
# Quick factories (optional)
# -----------------------------

def make_ficnn(in_dim: int, out_dim: int, hidden: Sequence[int] = (256, 128, 64)) -> FICNN:
    return FICNN(in_dim=in_dim, out_dim=out_dim, hidden_sizes=hidden, activation="relu")


def make_picnn(u_dim: int, v_dim: int, out_dim: int, hidden: Sequence[int] = (256, 128, 64)) -> PICNN:
    return PICNN(u_dim=u_dim, v_dim=v_dim, out_dim=out_dim, hidden_sizes=hidden, activation="relu")


def make_picnn_mtlsh(
    u_dim: int,
    v_dim: int,
    n_tasks: int,
    trunk: Sequence[int] = (256, 128),
    head: Sequence[int] = (128, 64),
) -> PICNN_MTLSH:
    return PICNN_MTLSH(u_dim=u_dim, v_dim=v_dim, n_tasks=n_tasks, trunk_sizes=trunk, head_sizes=head, activation="relu")
