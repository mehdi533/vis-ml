import torch
import torch.nn as nn
from typing import Callable, List, Tuple


class KendallMTLoss(nn.Module):
    """
    Multi-task regression loss with learned homoscedastic uncertainty.
    """

    def __init__(self, n_tasks: int = 4):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(n_tasks, dtype=torch.float32))

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        assert preds.shape == targets.shape, "preds and targets must have same shape"
        n_tasks = preds.shape[1]
        assert n_tasks == self.log_vars.shape[0], "Mismatch between n_tasks and log_vars"

        loss = 0.0
        for i in range(n_tasks):
            diff = preds[:, i] - targets[:, i]
            mse_i = torch.mean(diff ** 2)
            precision = torch.exp(-self.log_vars[i])
            loss_i = 0.5 * precision * mse_i + 0.5 * self.log_vars[i]
            loss = loss + loss_i
        return loss


LOSS_FACTORY = {
    "mse": nn.MSELoss,
    "l1": nn.L1Loss,
    "smooth_l1": nn.SmoothL1Loss,
    "huber": nn.HuberLoss,
    "kendall": KendallMTLoss,
}


def build_loss(
    loss_type: str,
    ce_weights: List[float],
    device: torch.device,
    out_dim: int,
) -> Tuple[Callable, List[torch.nn.Parameter]]:
    """
    Create a loss callable plus any learnable ancillary parameters.
    """
    if loss_type == "kendall":
        criterion = LOSS_FACTORY[loss_type](out_dim).to(device)

        def loss_fn(preds, targets):
            return criterion(preds, targets)

        loss_fn.log_vars = criterion.log_vars
        return loss_fn, list(criterion.parameters())

    base_cls = LOSS_FACTORY[loss_type]
    use_weights = bool(ce_weights)

    if use_weights:
        if len(ce_weights) != out_dim:
            raise ValueError(f"Expected {out_dim} loss weights, got {len(ce_weights)}.")
        w = torch.tensor(ce_weights, dtype=torch.float32, device=device)
        base_loss = base_cls(reduction="none")

        def loss_fn(preds, targets):
            loss = base_loss(preds, targets)
            return (loss * w).mean()

    else:
        base_loss = base_cls()

        def loss_fn(preds, targets):
            return base_loss(preds, targets)

    return loss_fn, []
