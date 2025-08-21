import torch
import torch.nn as nn

class loss(nn.Module):
    """Base class for loss functions."""
    def __init__(self, coverage, penalty=0):
        super().__init__()
        self.c = coverage
        self.l = penalty
        self.quantiles = [(1-coverage)/2, 1-(1-coverage)/2]
        self.q1 = self.quantiles[0]
        self.q2 = self.quantiles[1]
        
    def forward(self, y_pred, target):
        pass
    
class Tube_Loss(loss):
    """Tube (confidence) loss.

    Implements the piecewise "tube" loss with parameters:
    - coverage (q): target coverage level, taken from self.c
    - r: tube movement factor (default 0.5)
    - delta: recalibration coefficient on interval width (default 0.0)
    """
    def __init__(self, coverage, penalty=0, r: float = 0.5, delta: float = 0.03):
        super().__init__(coverage, penalty)
        self.r_value = float(r)
        self.delta_value = float(delta)

    def forward(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # y_true: ensure (N,)
        y_true = target[:, 0]

        # Model outputs: two bounds (sort to avoid bound crossing)
        raw_u = preds[:, 0]
        raw_l = preds[:, 1]
        f1 = torch.minimum(raw_l, raw_u)
        f2 = torch.maximum(raw_l, raw_u)

        # Tube parameters
        q = self.c  # target coverage (0.9 for 90%)
        r = torch.tensor(self.r_value, dtype=preds.dtype, device=preds.device)
        delta = torch.tensor(self.delta_value, dtype=preds.dtype, device=preds.device)

        # Define loss components
        c1 = (1 - q) * (f2 - y_true)
        c2 = (1 - q) * (y_true - f1)
        c3 = q * (f1 - y_true)
        c4 = q * (y_true - f2)

        # Tube loss logic
        condition1 = y_true > r * (f1 + f2)
        loss_part1 = torch.where(condition1, c1, c2)

        condition2 = f1 > y_true
        loss_part2 = torch.where(condition2, c3, c4)

        # Final loss selection
        in_interval = torch.logical_and(y_true <= f2, y_true >= f1)
        final_loss = torch.where(in_interval, loss_part1, loss_part2) + (delta * torch.abs(f1 - f2))

        return torch.mean(final_loss)
    
    
dict_loss = {"TUBE": Tube_Loss}

