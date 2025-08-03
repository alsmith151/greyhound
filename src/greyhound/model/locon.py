import re

import torch
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model
import numpy as np
from loguru import logger
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

def add_locon(model, rank=4, alpha=1.0, conv_select=None, ignore_layers=None, **lora_kwargs):
    """
    Add LoRA to attention and LoCon to selected Conv1D layers in a PyTorch model.
    
    Args:
        model: The model to modify
        rank: LoCon rank
        alpha: LoCon alpha scaling factor
        conv_select: Number of Conv1D layers to select (from the end)
        ignore_layers: List of layer name patterns/substrings to ignore.
                      Can be strings or regex patterns. Default is ["chromatin_head"]
        **lora_kwargs: Additional arguments for LoRA configuration
    """

    initial_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Set default ignore patterns
    if ignore_layers is None:
        ignore_layers = ["chromatin_head"]
    elif isinstance(ignore_layers, str):
        ignore_layers = [ignore_layers]

    # Apply LoRA to attention layers
    lora_config = LoraConfig(
        **lora_kwargs,
        task_type=TaskType.FEATURE_EXTRACTION,
        bias="none",
        modules_to_save=["chromatin_head"]
    )
    model = get_peft_model(model, lora_config)
    post_lora_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Find all Conv1d layers, excluding specified patterns
    def should_ignore_layer(name):
        for pattern in ignore_layers:
            if isinstance(pattern, str):
                if pattern in name:
                    return True
            else:
                # Assume it's a regex pattern
                if re.search(pattern, name):
                    return True
        return False

    conv_layers = [(name, module) for name, module in model.named_modules()
                   if isinstance(module, nn.Conv1d) and not should_ignore_layer(name)]

    if not conv_layers:
        print(f"No Conv1d layers found (excluding patterns: {ignore_layers}).")
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
    post_locon_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"Total parameters before LoRA and LoCon: {initial_params}")
    logger.info(f"Number of parameters after LoRA: {post_lora_params}")
    logger.info(f"Number of parameters after LoCon: {post_locon_params}")
    logger.info(f"Total trainable parameters after LoRA and LoCon: {trainable_params}")
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
                
                # Get the weights from down and up layers
                down_weight = module.down_layer.weight  # Shape: [rank, in_channels, kernel_size]
                up_weight = module.up_layer.weight      # Shape: [out_channels, rank, 1]
                
                # Compute the effective weight increment by matrix multiplication
                # We need to reshape and multiply the weights properly
                if down_weight.dim() == 3 and up_weight.dim() == 3:
                    # For Conv1d: multiply up_weight[:, :, 0] @ down_weight.view(rank, -1)
                    # Then reshape back to original conv weight shape
                    rank, in_channels, kernel_size = down_weight.shape
                    out_channels = up_weight.shape[0]
                    
                    # Flatten down_weight for matrix multiplication
                    down_flat = down_weight.view(rank, -1)  # [rank, in_channels * kernel_size]
                    up_flat = up_weight.squeeze(-1)         # [out_channels, rank]
                    
                    # Compute increment
                    increment_flat = torch.mm(up_flat, down_flat)  # [out_channels, in_channels * kernel_size]
                    increment = increment_flat.view(out_channels, in_channels, kernel_size)
                    
                    # Apply scaling and add to original weights
                    original.weight += increment * module.scale
                
            # Replace LoCon with merged original conv layer
            parent = model
            parts = name.split(".")
            for p in parts[:-1]:
                if p.isdigit():
                    parent = parent[int(p)]
                else:
                    parent = getattr(parent, p)
            last_part = parts[-1]
            if last_part.isdigit():
                parent[int(last_part)] = module.conv_layer
            else:
                setattr(parent, last_part, module.conv_layer)

    print("All LoRA and LoCon weights merged into the base model.")
    return model
