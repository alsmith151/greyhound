import torch
from transformers import EvalPrediction

from .losses import multinomial_loss, poisson_loss, poisson_multinomial_combined_loss


def compute_metrics(eval_pred: EvalPrediction) -> dict[str, float]:
    """
    Compute metrics for the model predictions.

    Just returns the loss metric for now.
    """
    predictions, labels = eval_pred
    predictions = torch.tensor(predictions, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.float32)

    poisson_loss_value = poisson_loss(predictions, labels)
    multinomial_loss_value = multinomial_loss(predictions, labels)
    combined_loss_value = poisson_multinomial_combined_loss(predictions, labels)
    return {
        "poisson_loss": poisson_loss_value.item(),
        "multinomial_loss": multinomial_loss_value.item(),
        "combined_loss": combined_loss_value.item(),
    }
