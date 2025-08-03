import os

from loguru import logger
from transformers import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
from copy import deepcopy

from .locon import LoCon, merge_locon


class SaveMergedModelCallback(TrainerCallback):
    def __init__(
        self,
        save_every: int = 1024,
        save_path_prefix: str = "model_weights/long_boi_lora",
    ):
        self.save_every = save_every
        self.save_path_prefix = save_path_prefix

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        from copy import deepcopy

        step = state.global_step
        if step > 0 and step % self.save_every == 0:
            model = deepcopy(kwargs["model"])
            merged_model = model.merge_and_unload()
            save_path = f"{self.save_path_prefix}_{step}"
            merged_model.save_pretrained(save_path)
            logger.info(f"✅ Merged model saved at step {step} → {save_path}")


class LoConMergeCallback(TrainerCallback):
    """
    Custom callback that saves merged LoCon models as pretrained models.
    """

    def __init__(self, save_merged_every: int = 1000):
        """
        Args:
            save_merged_every (int): Save merged model every N steps.
        """
        self.save_merged_every = save_merged_every

    def on_step_end(
        self, args, state: TrainerState, control: TrainerControl, model=None, **kwargs
    ):
        """
        Save merged model at specified intervals.
        """
        if model is None or state.global_step % self.save_merged_every != 0:
            return

        # Check if model has LoCon adapters
        has_locon = any(isinstance(module, LoCon) for module in model.modules())
        if not has_locon:
            return

        print(f"💾 Saving merged LoCon model at step {state.global_step}...")
        
        try:
            # Create a deep copy to avoid modifying the training model
            merged_model = deepcopy(model)
            merged_model = merge_locon(merged_model)
            
            # Get the base model without adapters
            if hasattr(merged_model, 'base_model'):
                base_model = merged_model.base_model
            elif hasattr(merged_model, 'model'):
                base_model = merged_model.model
            else:
                base_model = merged_model
            
            # Save only the base model as pretrained model
            merged_dir = os.path.join(args.output_dir, f"merged_model_step_{state.global_step}")
            base_model.save_pretrained(merged_dir)
            print(f"✅ Merged model saved to: {merged_dir}")
            
            # Explicitly delete the copy to free memory
            del merged_model, base_model
            
        except Exception as e:
            print(f"❌ Error saving merged model: {e}")

    def on_train_end(
        self, args, state: TrainerState, control: TrainerControl, model=None, **kwargs
    ):
        """
        Save final merged model at end of training.
        """
        if model is None:
            return

        # Check if model has LoCon adapters
        has_locon = any(isinstance(module, LoCon) for module in model.modules())
        if not has_locon:
            return

        print("🏁 Saving final merged LoCon model...")
        
        try:
            # Create a deep copy to avoid modifying the training model
            merged_model = deepcopy(model)
            merged_model = merge_locon(merged_model)
            
            # Get the base model without adapters
            if hasattr(merged_model, 'base_model'):
                base_model = merged_model.base_model
            elif hasattr(merged_model, 'model'):
                base_model = merged_model.model
            else:
                base_model = merged_model
            
            # Save only the base model as pretrained model
            final_dir = os.path.join(args.output_dir, "final_merged_model")
            base_model.save_pretrained(final_dir)
            print(f"✅ Final merged model saved to: {final_dir}")
            
            # Explicitly delete the copy to free memory
            del merged_model, base_model
            
        except Exception as e:
            print(f"❌ Error saving final merged model: {e}")
