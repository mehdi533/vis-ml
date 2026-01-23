from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn


class MLP(nn.Module):
    """Simple feed-forward network with configurable hidden layers."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_sizes: Optional[Sequence[int]] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        hidden_sizes = hidden_sizes or [256, 128, 64, 64]
        layers = []
        last_dim = in_dim

        for h in hidden_sizes:
            layers.append(nn.Linear(last_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            last_dim = h

        layers.append(nn.Linear(last_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FeatureAttention(nn.Module):
    """Feature-wise attention gating for tabular inputs."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: Optional[int] = None,
        temperature: float = 1.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        if hidden_dim is None:
            self.net = nn.Linear(in_dim, in_dim)
        else:
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, in_dim),
            )
        self.temperature = float(temperature)
        self.last_attn: Optional[torch.Tensor] = None
        self.last_logits: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.net(x)
        attn = torch.softmax(logits / self.temperature, dim=-1)
        self.last_attn = attn.detach()
        self.last_logits = logits.detach()
        return x * attn, attn


class KANLinear(nn.Module):
    """Piecewise-linear spline layer (KAN-style) with per-edge 1D functions."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        grid_size: int = 8,
        grid_min: float = -1.0,
        grid_max: float = 1.0,
        bias: bool = True,
    ):
        super().__init__()
        if grid_size < 2:
            raise ValueError("grid_size must be >= 2.")
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.grid_size = grid_size
        self.grid_min = float(grid_min)
        self.grid_max = float(grid_max)
        knots = torch.linspace(self.grid_min, self.grid_max, grid_size)
        self.register_buffer("knots", knots)
        self.coeffs = nn.Parameter(torch.randn(out_dim, in_dim, grid_size) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_dim)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clamp(self.grid_min, self.grid_max).contiguous()
        eps = 1e-6
        idx = torch.bucketize(x, self.knots) - 1
        idx = idx.clamp(0, self.grid_size - 2)
        left = self.knots[idx]
        right = self.knots[idx + 1]
        t = (x - left) / (right - left + eps)

        coeffs = self.coeffs.unsqueeze(0).expand(x.size(0), -1, -1, -1)
        idx_exp = idx.unsqueeze(1).unsqueeze(-1).expand(-1, self.out_dim, -1, 1)
        idx_next = (idx + 1).unsqueeze(1).unsqueeze(-1).expand(-1, self.out_dim, -1, 1)
        c_left = torch.gather(coeffs, 3, idx_exp).squeeze(-1)
        c_right = torch.gather(coeffs, 3, idx_next).squeeze(-1)
        out = (c_left * (1.0 - t).unsqueeze(1) + c_right * t.unsqueeze(1)).sum(dim=2)
        if self.bias is not None:
            out = out + self.bias
        return out

    def edge_params(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.knots.detach().cpu(), self.coeffs.detach().cpu()


class KANBlock(nn.Module):
    """Stack of KANLinear layers with optional dropout."""

    def __init__(
        self,
        in_dim: int,
        hidden_sizes: Sequence[int],
        dropout: float = 0.0,
        grid_size: int = 8,
        grid_min: float = -1.0,
        grid_max: float = 1.0,
    ):
        super().__init__()
        layers = []
        last = in_dim
        for h in hidden_sizes:
            layers.append(
                KANLinear(
                    last,
                    h,
                    grid_size=grid_size,
                    grid_min=grid_min,
                    grid_max=grid_max,
                )
            )
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            last = h
        self.net = nn.Sequential(*layers) if layers else nn.Identity()
        self.out_dim = last

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MTLSharedHeads(nn.Module):
    """Multi-task model with a shared trunk and separate heads for each target."""

    def __init__(
        self,
        in_dim: int,
        n_tasks: int = 3,
        shared_sizes: Optional[Sequence[int]] = None,
        head_sizes: Optional[Sequence[int]] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        shared_sizes = shared_sizes or [256, 128]
        head_sizes = head_sizes or [128, 64]

        shared_layers = []
        last = in_dim
        for h in shared_sizes:
            shared_layers += [nn.Linear(last, h), nn.ReLU(), nn.Dropout(dropout)]
            last = h
        self.shared = nn.Sequential(*shared_layers)

        heads = []
        for _ in range(n_tasks):
            layers = []
            last_h = last
            for h in head_sizes:
                layers += [nn.Linear(last_h, h), nn.ReLU(), nn.Dropout(dropout)]
                last_h = h
            layers.append(nn.Linear(last_h, 1))
            heads.append(nn.Sequential(*layers))

        self.heads = nn.ModuleList(heads)
        self.n_tasks = n_tasks

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.shared(x)
        outs = [head(h) for head in self.heads]
        return torch.cat(outs, dim=1)


@dataclass(frozen=True)
class SharedGroupSpec:
    """Defines additional shared layers for a subset of heads."""

    head_indices: Sequence[int]
    hidden_sizes: Sequence[int]


class MTLGroupedSharedHeads(nn.Module):
    """Shared trunk with optional grouped shared blocks and per-task heads."""

    def __init__(
        self,
        in_dim: int,
        n_tasks: int,
        shared_sizes: Optional[Sequence[int]] = None,
        head_sizes: Optional[Sequence[int]] = None,
        dropout: float = 0.0,
        group_shared_configs: Optional[Sequence[SharedGroupSpec]] = None,
    ):
        super().__init__()
        shared_sizes = shared_sizes or [256, 128]
        head_sizes = head_sizes or [128, 64]

        shared_layers = []
        last = in_dim
        for h in shared_sizes:
            shared_layers += [nn.Linear(last, h), nn.ReLU(), nn.Dropout(dropout)]
            last = h
        self.shared = nn.Sequential(*shared_layers)

        base_dim = last if shared_sizes else in_dim
        head_input_dims = [base_dim] * n_tasks

        self.group_blocks = nn.ModuleList()
        self.group_block_indices: List[Tuple[int, ...]] = []
        if group_shared_configs:
            for config in group_shared_configs:
                if not config.hidden_sizes or not config.head_indices:
                    continue

                indices = tuple(config.head_indices)
                for idx in indices:
                    if idx < 0 or idx >= n_tasks:
                        raise IndexError(f"Head index {idx} out of range [0, {n_tasks}).")

                current_dim = head_input_dims[indices[0]]
                if any(head_input_dims[idx] != current_dim for idx in indices):
                    raise ValueError("Heads sharing extra layers must have the same input dimension.")

                block, out_dim = self._build_shared_block(current_dim, config.hidden_sizes, dropout)
                self.group_blocks.append(block)
                self.group_block_indices.append(indices)
                for idx in indices:
                    head_input_dims[idx] = out_dim

        self.heads = nn.ModuleList()
        for dim in head_input_dims:
            self.heads.append(self._make_head(dim, head_sizes, 0.1))

        self.n_tasks = n_tasks

    def _build_shared_block(self, in_dim: int, hidden_sizes: Sequence[int], dropout: float):
        layers = []
        last = in_dim
        for size in hidden_sizes:
            layers += [nn.Linear(last, size), nn.ReLU(), nn.Dropout(dropout)]
            last = size
        return nn.Sequential(*layers), last

    def _make_head(self, in_dim: int, head_sizes: Sequence[int], dropout: float):
        layers = []
        last = in_dim
        for h in head_sizes:
            layers += [nn.Linear(last, h), nn.ReLU(), nn.Dropout(dropout)]
            last = h
        layers.append(nn.Linear(last, 1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.shared(x)
        head_inputs = [h] * self.n_tasks

        for block, indices in zip(self.group_blocks, self.group_block_indices):
            for idx in indices:
                head_inputs[idx] = block(head_inputs[idx])

        outs = [self.heads[i](head_inputs[i]) for i in range(self.n_tasks)]
        return torch.cat(outs, dim=1)


class MTLGroupedSharedHeadsAttn(nn.Module):
    """MTLGroupedSharedHeads with feature-wise attention before the shared trunk."""

    def __init__(
        self,
        in_dim: int,
        n_tasks: int,
        shared_sizes: Optional[Sequence[int]] = None,
        head_sizes: Optional[Sequence[int]] = None,
        dropout: float = 0.0,
        group_shared_configs: Optional[Sequence[SharedGroupSpec]] = None,
        attention_hidden_dim: Optional[int] = None,
        attention_temperature: float = 1.0,
        attention_dropout: float = 0.0,
    ):
        super().__init__()
        shared_sizes = shared_sizes or [256, 128]
        head_sizes = head_sizes or [128, 64]

        self.attention = FeatureAttention(
            in_dim,
            hidden_dim=attention_hidden_dim,
            temperature=attention_temperature,
            dropout=attention_dropout,
        )
        self.last_attn_input: Optional[torch.Tensor] = None

        shared_layers = []
        last = in_dim
        for h in shared_sizes:
            shared_layers += [nn.Linear(last, h), nn.ReLU(), nn.Dropout(dropout)]
            last = h
        self.shared = nn.Sequential(*shared_layers)

        base_dim = last if shared_sizes else in_dim
        head_input_dims = [base_dim] * n_tasks

        self.group_blocks = nn.ModuleList()
        self.group_block_indices: List[Tuple[int, ...]] = []
        if group_shared_configs:
            for config in group_shared_configs:
                if not config.hidden_sizes or not config.head_indices:
                    continue

                indices = tuple(config.head_indices)
                for idx in indices:
                    if idx < 0 or idx >= n_tasks:
                        raise IndexError(f"Head index {idx} out of range [0, {n_tasks}).")

                current_dim = head_input_dims[indices[0]]
                if any(head_input_dims[idx] != current_dim for idx in indices):
                    raise ValueError("Heads sharing extra layers must have the same input dimension.")

                block, out_dim = self._build_shared_block(current_dim, config.hidden_sizes, dropout)
                self.group_blocks.append(block)
                self.group_block_indices.append(indices)
                for idx in indices:
                    head_input_dims[idx] = out_dim

        self.heads = nn.ModuleList()
        for dim in head_input_dims:
            self.heads.append(self._make_head(dim, head_sizes, 0.1))

        self.n_tasks = n_tasks

    def _build_shared_block(self, in_dim: int, hidden_sizes: Sequence[int], dropout: float):
        layers = []
        last = in_dim
        for size in hidden_sizes:
            layers += [nn.Linear(last, size), nn.ReLU(), nn.Dropout(dropout)]
            last = size
        return nn.Sequential(*layers), last

    def _make_head(self, in_dim: int, head_sizes: Sequence[int], dropout: float):
        layers = []
        last = in_dim
        for h in head_sizes:
            layers += [nn.Linear(last, h), nn.ReLU(), nn.Dropout(dropout)]
            last = h
        layers.append(nn.Linear(last, 1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        x_attn, attn = self.attention(x)
        self.last_attn_input = attn
        h = self.shared(x_attn)
        head_inputs = [h] * self.n_tasks

        for block, indices in zip(self.group_blocks, self.group_block_indices):
            for idx in indices:
                head_inputs[idx] = block(head_inputs[idx])

        outs = [self.heads[i](head_inputs[i]) for i in range(self.n_tasks)]
        out = torch.cat(outs, dim=1)
        if return_attn:
            return out, attn
        return out


class MTLGroupedSharedHeadsKANShared(nn.Module):
    """KAN shared trunk + KAN group blocks + MLP heads."""

    def __init__(
        self,
        in_dim: int,
        n_tasks: int,
        shared_sizes: Optional[Sequence[int]] = None,
        head_sizes: Optional[Sequence[int]] = None,
        dropout: float = 0.0,
        group_shared_configs: Optional[Sequence[SharedGroupSpec]] = None,
        kan_grid_size: int = 8,
        kan_grid_min: float = -1.0,
        kan_grid_max: float = 1.0,
    ):
        super().__init__()
        shared_sizes = shared_sizes or [256, 128]
        head_sizes = head_sizes or [128, 64]

        self.shared = KANBlock(
            in_dim,
            shared_sizes,
            dropout=dropout,
            grid_size=kan_grid_size,
            grid_min=kan_grid_min,
            grid_max=kan_grid_max,
        )

        base_dim = self.shared.out_dim if shared_sizes else in_dim
        head_input_dims = [base_dim] * n_tasks

        self.group_blocks = nn.ModuleList()
        self.group_block_indices: List[Tuple[int, ...]] = []
        if group_shared_configs:
            for config in group_shared_configs:
                if not config.hidden_sizes or not config.head_indices:
                    continue

                indices = tuple(config.head_indices)
                for idx in indices:
                    if idx < 0 or idx >= n_tasks:
                        raise IndexError(f"Head index {idx} out of range [0, {n_tasks}).")

                current_dim = head_input_dims[indices[0]]
                if any(head_input_dims[idx] != current_dim for idx in indices):
                    raise ValueError("Heads sharing extra layers must have the same input dimension.")

                block = KANBlock(
                    current_dim,
                    config.hidden_sizes,
                    dropout=dropout,
                    grid_size=kan_grid_size,
                    grid_min=kan_grid_min,
                    grid_max=kan_grid_max,
                )
                self.group_blocks.append(block)
                self.group_block_indices.append(indices)
                for idx in indices:
                    head_input_dims[idx] = block.out_dim

        self.heads = nn.ModuleList()
        for dim in head_input_dims:
            self.heads.append(self._make_head(dim, head_sizes, 0.1))

        self.n_tasks = n_tasks

    def _make_head(self, in_dim: int, head_sizes: Sequence[int], dropout: float):
        layers = []
        last = in_dim
        for h in head_sizes:
            layers += [nn.Linear(last, h), nn.ReLU(), nn.Dropout(dropout)]
            last = h
        layers.append(nn.Linear(last, 1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.shared(x)
        head_inputs = [h] * self.n_tasks

        for block, indices in zip(self.group_blocks, self.group_block_indices):
            for idx in indices:
                head_inputs[idx] = block(head_inputs[idx])

        outs = [self.heads[i](head_inputs[i]) for i in range(self.n_tasks)]
        return torch.cat(outs, dim=1)


class MTLGroupedSharedHeadsKAN(nn.Module):
    """Fully KAN: shared trunk + group blocks + KAN heads."""

    def __init__(
        self,
        in_dim: int,
        n_tasks: int,
        shared_sizes: Optional[Sequence[int]] = None,
        head_sizes: Optional[Sequence[int]] = None,
        dropout: float = 0.0,
        group_shared_configs: Optional[Sequence[SharedGroupSpec]] = None,
        kan_grid_size: int = 8,
        kan_grid_min: float = -1.0,
        kan_grid_max: float = 1.0,
    ):
        super().__init__()
        shared_sizes = shared_sizes or [256, 128]
        head_sizes = head_sizes or [128, 64]

        self.shared = KANBlock(
            in_dim,
            shared_sizes,
            dropout=dropout,
            grid_size=kan_grid_size,
            grid_min=kan_grid_min,
            grid_max=kan_grid_max,
        )

        base_dim = self.shared.out_dim if shared_sizes else in_dim
        head_input_dims = [base_dim] * n_tasks

        self.group_blocks = nn.ModuleList()
        self.group_block_indices: List[Tuple[int, ...]] = []
        if group_shared_configs:
            for config in group_shared_configs:
                if not config.hidden_sizes or not config.head_indices:
                    continue

                indices = tuple(config.head_indices)
                for idx in indices:
                    if idx < 0 or idx >= n_tasks:
                        raise IndexError(f"Head index {idx} out of range [0, {n_tasks}).")

                current_dim = head_input_dims[indices[0]]
                if any(head_input_dims[idx] != current_dim for idx in indices):
                    raise ValueError("Heads sharing extra layers must have the same input dimension.")

                block = KANBlock(
                    current_dim,
                    config.hidden_sizes,
                    dropout=dropout,
                    grid_size=kan_grid_size,
                    grid_min=kan_grid_min,
                    grid_max=kan_grid_max,
                )
                self.group_blocks.append(block)
                self.group_block_indices.append(indices)
                for idx in indices:
                    head_input_dims[idx] = block.out_dim

        self.heads = nn.ModuleList()
        for dim in head_input_dims:
            self.heads.append(
                self._make_kan_head(
                    dim,
                    head_sizes,
                    dropout=dropout,
                    grid_size=kan_grid_size,
                    grid_min=kan_grid_min,
                    grid_max=kan_grid_max,
                )
            )

        self.n_tasks = n_tasks

    def _make_kan_head(
        self,
        in_dim: int,
        head_sizes: Sequence[int],
        dropout: float,
        grid_size: int,
        grid_min: float,
        grid_max: float,
    ):
        layers = []
        last = in_dim
        for h in head_sizes:
            layers.append(
                KANLinear(
                    last,
                    h,
                    grid_size=grid_size,
                    grid_min=grid_min,
                    grid_max=grid_max,
                )
            )
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            last = h
        layers.append(
            KANLinear(
                last,
                1,
                grid_size=grid_size,
                grid_min=grid_min,
                grid_max=grid_max,
            )
        )
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.shared(x)
        head_inputs = [h] * self.n_tasks

        for block, indices in zip(self.group_blocks, self.group_block_indices):
            for idx in indices:
                head_inputs[idx] = block(head_inputs[idx])

        outs = [self.heads[i](head_inputs[i]) for i in range(self.n_tasks)]
        return torch.cat(outs, dim=1)


MODEL_FACTORY = {
    "MLP": MLP,
    "MTLSH": MTLSharedHeads,
    "MTLGSH": MTLGroupedSharedHeads,
    "MTLGSH_ATT": MTLGroupedSharedHeadsAttn,
    "MTLGSH_KAN_SHARED": MTLGroupedSharedHeadsKANShared,
    "MTLGSH_KAN": MTLGroupedSharedHeadsKAN,
}

MODEL_CATALOG = {name: cls.__name__ for name, cls in MODEL_FACTORY.items()}


def list_models():
    return sorted(MODEL_CATALOG.keys())


def create_model(
    model_type: str,
    in_dim: int,
    out_dim: int,
    device=None,
    group_head_indices: Optional[Sequence[Sequence[int]]] = None,
    kan_grid_size: int = 8,
    kan_grid_min: float = -1.0,
    kan_grid_max: float = 1.0,
    attention_hidden_dim: Optional[int] = None,
    attention_temperature: float = 1.0,
    attention_dropout: float = 0.0,
):
    if model_type not in {"MTLGSH", "MTLGSH_ATT", "MTLGSH_KAN_SHARED", "MTLGSH_KAN"}:
        model = MODEL_FACTORY[model_type](in_dim, out_dim)
    else:
        if group_head_indices is None:
            group_head_indices = [
                [0, 1, 2, 3],
                [4, 5, 6, 7],
            ]
        group_shared_configs = [
            SharedGroupSpec(head_indices=indices, hidden_sizes=[128, 64])
            for indices in group_head_indices
        ]
        if model_type == "MTLGSH":
            model = MTLGroupedSharedHeads(
                in_dim=in_dim,
                n_tasks=out_dim,
                shared_sizes=[256, 128],
                head_sizes=[64, 32],
                dropout=0,
                group_shared_configs=group_shared_configs,
            )
        elif model_type == "MTLGSH_ATT":
            model = MTLGroupedSharedHeadsAttn(
                in_dim=in_dim,
                n_tasks=out_dim,
                shared_sizes=[256, 128],
                head_sizes=[64, 32],
                dropout=0,
                group_shared_configs=group_shared_configs,
                attention_hidden_dim=attention_hidden_dim,
                attention_temperature=attention_temperature,
                attention_dropout=attention_dropout,
            )
        else:
            model = MODEL_FACTORY[model_type](
                in_dim=in_dim,
                n_tasks=out_dim,
                shared_sizes=[256, 128],
                head_sizes=[64, 32],
                dropout=0,
                group_shared_configs=group_shared_configs,
                kan_grid_size=kan_grid_size,
                kan_grid_min=kan_grid_min,
                kan_grid_max=kan_grid_max,
            )

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, device
