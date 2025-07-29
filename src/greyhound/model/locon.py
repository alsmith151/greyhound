import re

import torch
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model


class LoCon(nn.Module):
    """
    Custom LoCon adapter for Conv1d layers.
    """

    def __init__(self, conv_layer, rank=4, alpha=1.0):
        super().__init__()
        self.conv_layer = conv_layer
        self.rank = rank
        self.alpha = alpha
        in_channels = conv_layer.in_channels
        out_channels = conv_layer.out_channels
        kernel_size = conv_layer.kernel_size[0]

        # Low-rank adapters
        self.down_layer = nn.Conv1d(in_channels, rank, kernel_size=1, bias=False)
        self.up_layer = nn.Conv1d(rank, out_channels, kernel_size=1, bias=False)
        self.scale = alpha / rank

    def forward(self, x):
        increment = self.up_layer(self.down_layer(x)) * self.scale
        return self.conv_layer(x) + increment


def add_locon(model, rank=4, alpha=1.0, conv_select=None, **lora_kwargs):
    """
    Add LoRA to attention and LoCon to selected Conv1D layers.
    """
    # Apply LoRA to attention layers
    lora_config = LoraConfig(
        **lora_kwargs,
        target_modules=r'.*to_q|.*to_v',
        task_type=TaskType.FEATURE_EXTRACTION,
        bias="none",
    )

    model = get_peft_model(model, lora_config)

    # Find Conv1d layers
    conv_layers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv1d):
            conv_layers.append((name, module))

    if conv_select is None:
        conv_select = len(conv_layers)

    if conv_select > len(conv_layers):
        raise ValueError(f"conv_select must be <= {len(conv_layers)}")

    locon_layers = conv_layers[-conv_select:]
    conv1_tune = conv_select == len(conv_layers)

    # Replace Conv1d layers with LoCon-wrapped versions
    for name, conv in locon_layers:
        parent = model
        parts = name.split(".")
        for p in parts[:-1]:
            parent = getattr(parent, p)
        setattr(parent, parts[-1], LoCon(conv, rank=rank, alpha=alpha))

    if conv1_tune and conv_layers:
        conv1_name, conv1 = conv_layers[0]
        conv1.trainable = True

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"params added/unfrozen by locon: {trainable_params}")

    return model


def merge_locon(model):
    """
    Merge LoRA and LoCon weights into the base model.
    """
    for name, module in model.named_modules():
        # Merge LoRA using PEFT built-in utilities
        if hasattr(module, "merge_and_unload"):
            module.merge_and_unload()

        # Merge LoCon manually
        if isinstance(module, LoCon):
            with torch.no_grad():
                original = module.conv_layer
                increment = (
                    module.up_layer(
                        module.down_layer.weight.transpose(0, 1).unsqueeze(-1)
                    )
                    * module.scale
                )
                original.weight += increment.squeeze(-1)
            # Replace LoCon with merged original conv layer
            parent = model
            parts = name.split(".")
            for p in parts[:-1]:
                parent = getattr(parent, p)
            setattr(parent, parts[-1], module.conv_layer)

    print("All LoRA and LoCon weights merged into the base model.")
    return model
