from peft import LoraConfig, TaskType


def create_lora_config() -> LoraConfig:
    """
    Create optimal LoRA configuration for the Greyhound genomics model.
    
    This configuration is optimized for:
    - Genomics sequence data
    - Conv1d and Linear layer support
    - Memory efficiency
    - Training stability
    
    Returns:
        LoraConfig: Optimized LoRA configuration
    """
    
    config = LoraConfig(
        # Core LoRA parameters - optimized for genomics
        r=128,                    # Balanced rank for expressivity vs efficiency
        lora_alpha=256,          # 2x rank following best practices
        lora_dropout=0.1,       # Moderate dropout for regularization
        
        # Target modules - comprehensive coverage
        target_modules=[
            # Transformer attention layers (highest priority)
            "transformer.*.*.fn.1.mha.Wqkv",      # Query, Key, Value projections
            "transformer.*.*.fn.1.mha.out_proj",   # Attention output projection
            
            # Feed-forward network layers
            "transformer.*.*.fn.1.0",              # First FFN linear layer
            "transformer.*.*.fn.1.4",              # Second FFN linear layer
            
            # Convolutional layers (Conv1d LoRA supported)
            "borzoi.conv_dna.conv_layer",          # Initial DNA sequence processing
            "borzoi.res_tower.*.conv_layer",       # Residual tower convolutions
            "borzoi.horizontal_conv0.conv_layer",  # Horizontal feature processing
            "borzoi.horizontal_conv1.conv_layer",  # Horizontal feature processing
            
            
            # Final processing layers
            "borzoi.final_joined_convs.0.conv_layer",  # Pre-output convolution
            
            # Trained output heads only (exclude untrained chromatin_head)
            "borzoi.human_head",               # Human prediction head
            "borzoi.mouse_head",               # Mouse prediction head
        ],
        
        # Advanced LoRA settings
        bias="none",                    # Don't adapt bias terms for stability
        task_type=TaskType.FEATURE_EXTRACTION,  # Genomics feature extraction
        inference_mode=False,           # Training mode
        use_rslora=True,               # Rank-Stabilized LoRA for better training
        
        # Modules to train fully (not with LoRA)
        modules_to_save=["chromatin_head"],  # Untrained head needs full gradients
    )
    
    return config

