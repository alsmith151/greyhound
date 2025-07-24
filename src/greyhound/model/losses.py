import torch
import torch.nn.functional as F

def poisson_multinomial_combined_loss(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    total_weight: float = 0.2,
    epsilon: float = 1e-6,
    rescale: bool = False,
) -> torch.Tensor:
    """
    Combined Poisson + Multinomial loss:
    - Poisson loss for total counts.
    - Multinomial loss for positional distribution.

    Args:
        y_pred (Tensor): Predicted (batch_size, seq_len).
        y_true (Tensor): Ground truth (batch_size, seq_len).
        total_weight (float): Weight of Poisson term.
        epsilon (float): To prevent log(0).
        rescale (bool): If True, rescale to normalize loss scale.

    Returns:
        Mean loss (scalar).
    """
    y_true = y_true + epsilon
    y_pred = y_pred + epsilon

    s_true = y_true.sum(dim=1, keepdim=True)
    s_pred = y_pred.sum(dim=1, keepdim=True)

    # Poisson loss (total count)
    poisson_term = F.poisson_nll_loss(s_pred, s_true, log_input=False, reduction="mean")

    # Multinomial loss (distribution)
    p_pred = y_pred / s_pred
    multinomial_term = -torch.sum(y_true * torch.log(p_pred), dim=1).mean()

    # Combined loss
    loss = multinomial_term + total_weight * poisson_term

    if rescale:
        loss *= 2 / (1 + total_weight)

    return loss


def multinomial_loss(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    total_weight: float = 0.2,  # unused, for compatibility
    epsilon: float = 1e-6,
    rescale: bool = False,
) -> torch.Tensor:
    """
    Multinomial loss: compares predicted vs true positional distributions.

    Args:
        y_pred (Tensor): Predicted (batch_size, seq_len).
        y_true (Tensor): Ground truth (batch_size, seq_len).
        epsilon (float): To prevent log(0).
        rescale (bool): If True, rescale to match poisson_multinomial scale.

    Returns:
        Mean loss (scalar).
    """
    y_true = y_true + epsilon
    y_pred = y_pred + epsilon

    p_pred = y_pred / y_pred.sum(dim=1, keepdim=True)
    multinomial_term = -torch.sum(y_true * torch.log(p_pred), dim=1).mean()

    if rescale:
        multinomial_term *= 2 / (1 + total_weight)

    return multinomial_term


def poisson_loss(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    total_weight: float = 0.2,  # unused, for compatibility
    epsilon: float = 1e-6,
    rescale: bool = False,
) -> torch.Tensor:
    """
    Poisson loss: compares total predicted vs true counts.

    Args:
        y_pred (Tensor): Predicted (batch_size, seq_len).
        y_true (Tensor): Ground truth (batch_size, seq_len).
        epsilon (float): To prevent log(0).
        rescale (bool): If True, rescale to match poisson_multinomial scale.

    Returns:
        Mean loss (scalar).
    """
    y_true = y_true + epsilon
    y_pred = y_pred + epsilon

    s_true = y_true.sum(dim=1, keepdim=True)
    s_pred = y_pred.sum(dim=1, keepdim=True)

    poisson_term = F.poisson_nll_loss(s_pred, s_true, log_input=False, reduction="mean")

    if rescale:
        poisson_term *= 2 / (1 + total_weight)

    return poisson_term
