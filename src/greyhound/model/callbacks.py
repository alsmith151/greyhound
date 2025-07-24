from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from loguru import logger


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
