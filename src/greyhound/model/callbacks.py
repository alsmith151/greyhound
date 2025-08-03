import os

from loguru import logger
from transformers import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
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
    Custom callback that merges LoCon weights before saving checkpoints.
    This ensures that saved models contain the merged weights rather than the adapter structure.
    """

    def __init__(self, save_merged_only=True):
        """
        Args:
            save_merged_only (bool): If True, only save the merged model.
                                   If False, save both adapter and merged versions.
        """
        self.save_merged_only = save_merged_only
        self.original_model = None

    def on_save(
        self, args, state: TrainerState, control: TrainerControl, model=None, **kwargs
    ):
        """
        Called before saving the model. Merges LoCon weights and optionally saves both versions.
        """
        if model is None:
            return

        # Check if model has LoCon adapters
        has_locon = any(isinstance(module, LoCon) for module in model.modules())

        if not has_locon:
            print("No LoCon adapters found in model, proceeding with normal save.")
            return

        print(f"LoConMergeCallback: Merging LoCon weights before saving checkpoint...")

        # Store original model if we need to restore it later
        if not self.save_merged_only:
            self.original_model = deepcopy(model)

        # Merge the LoCon weights into the model
        try:
            model = merge_locon(model)
            print("✅ LoCon weights merged successfully")

            if not self.save_merged_only:
                # Save the merged model with a special suffix
                checkpoint_dir = os.path.join(
                    args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
                )
                merged_dir = os.path.join(checkpoint_dir, "merged_model")
                os.makedirs(merged_dir, exist_ok=True)

                # Save the merged model
                model.save_pretrained(merged_dir)
                print(f"📁 Merged model saved to: {merged_dir}")

                # Restore the original model with adapters for continued training
                model.load_state_dict(self.original_model.state_dict())
                print("🔄 Original model with adapters restored for continued training")

        except Exception as e:
            print(f"❌ Error during LoCon merge: {e}")
            print("Proceeding with normal save (adapter model)")

    def on_train_end(
        self, args, state: TrainerState, control: TrainerControl, model=None, **kwargs
    ):
        """
        Called at the end of training. Ensures final model is merged.
        """
        if model is None:
            return

        # Check if model has LoCon adapters
        has_locon = any(isinstance(module, LoCon) for module in model.modules())

        if has_locon:
            print("🏁 Training completed. Merging LoCon weights in final model...")
            model = merge_locon(model)
            print("✅ Final model LoCon weights merged")
