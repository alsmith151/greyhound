import re

import torch
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model
import numpy as np

import torch.nn.functional as F

class LoCon(nn.Module):
    """
    PyTorch implementation of LoCon adapter matching Keras behavior.
    """

    def __init__(self, conv_layer, rank=4, alpha=1.0):
        super().__init__()
        self.conv_layer = conv_layer
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank

        # Extract original Conv1D layer parameters
        in_channels = conv_layer.in_channels
        out_channels = conv_layer.out_channels
        kernel_size = conv_layer.kernel_size[0]
        stride = conv_layer.stride[0]
        dilation = conv_layer.dilation[0]
        padding = conv_layer.padding  

        # Low-rank adapters
        self.down_layer = nn.Conv1d(
            in_channels,
            rank,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=False
        )
        self.up_layer = nn.Conv1d(
            rank,
            out_channels,
            kernel_size=1,
            stride=stride,
            padding=0,  # PyTorch equivalent of "valid" (Keras uses "same" with kernel 1)
            bias=False
        )

        # Init weights to match Keras version
        nn.init.kaiming_uniform_(self.down_layer.weight, a=np.sqrt(5))
        nn.init.zeros_(self.up_layer.weight)

        # Freeze the original conv layer
        for param in self.conv_layer.parameters():
            param.requires_grad = False

    def forward(self, x):
        original_output = self.conv_layer(x)
        lora_output = self.up_layer(self.down_layer(x)) * self.scale
        return original_output + lora_output


def replace_module(model, module_path, new_module):
    """
    Replaces a module given a dotted path, e.g., 'encoder.layers.0.conv'.
    """
    parts = module_path.split(".")
    parent = model
    for p in parts[:-1]:
        if p.isdigit():
            parent = parent[int(p)]
        else:
            parent = getattr(parent, p)
    last = parts[-1]
    if last.isdigit():
        parent[int(last)] = new_module
    else:
        setattr(parent, last, new_module)

def add_locon(model, rank=4, alpha=1.0, conv_select=None, **lora_kwargs):
    """
    Add LoRA to attention and LoCon to selected Conv1D layers in a PyTorch model.
    """

    # Apply LoRA to attention layers
    lora_config = LoraConfig(
        **lora_kwargs,
        target_modules=r'.*to_q|.*to_v',
        task_type=TaskType.FEATURE_EXTRACTION,
        bias="none",
        modules_to_save=["chromatin_head"]
    )
    model = get_peft_model(model, lora_config)

    # Find all Conv1d layers
    conv_layers = [(name, module) for name, module in model.named_modules() if isinstance(module, nn.Conv1d)]

    if not conv_layers:
        print("No Conv1d layers found.")
        return model

    if conv_select is None:
        conv_select = len(conv_layers)

    if conv_select > len(conv_layers):
        raise ValueError(f"conv_select must be <= {len(conv_layers)}")

    locon_layers = conv_layers[-conv_select:]

    # Replace selected Conv1d layers with LoCon adapters
    for name, conv in locon_layers:
        wrapped = LoCon(conv, rank=rank, alpha=alpha)
        replace_module(model, name, wrapped)

    # Unfreeze the first Conv1d layer if all were selected
    if conv_select == len(conv_layers):
        first_conv = locon_layers[0][1]
        for param in first_conv.parameters():
            param.requires_grad = True

    # Report trainable parameters
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
