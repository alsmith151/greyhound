import os

os.environ["HF_HOME"] = (
    "/ceph/project/milne_group/asmith/Projects/2025-05-30-long-boi/cache"
)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import pathlib
import subprocess
import sys
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import pybigtools
import ray
import torch
import tqdm
from borzoi_pytorch import Borzoi
from loguru import logger
from peft import LoraConfig, get_peft_model
from pydantic import BaseModel, Field
from transformers import (
    EarlyStoppingCallback,
    EvalPrediction,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

import wandb

os.chdir("/home/a/asmith/project_milne_group/Projects/2025-05-30-long-boi/")
sys.path.append("/ceph/project/milne_group/asmith/Projects/2025-05-30-long-boi")
from src.datasets import (
    ChromatinDataset,
    GenomeIntervalDataset,
    test_filter,
    toy_filter,
    train_filter,
    val_filter,
)
from src.loss import multinomial_loss, poisson_loss, poisson_multinomial_combined_loss
from src.metrics import compute_metrics
from src.model import LongBoiConfig, LongBoi
from src.callbacks import SaveMergedModelCallback

BED_FILE = "/ceph/project/milne_group/asmith/Projects/2025-05-30-long-boi/data/sequences_human.bed.gz"
FASTA_FILE = "/project/milne_group/shared/seqnado_reference/hg38/UCSC/sequence/hg38.fa"
BIGWIG_DIR = "/ceph/project/milne_group/asmith/Projects/2025-05-30-long-boi/data/bigwigs_rpkm_updated"


########################
# DATASET PREPARATION
########################

genome_datasets = dict()
for dataset in {"train": train_filter, "val": val_filter, "test": test_filter}.items():
    name, filter_func = dataset

    filter_func = partial(
        filter_func,
        test_fold=3,  # Adjust these values as needed
        val_fold=4,  # Adjust these values as needed
    )

    genome_datasets[name] = GenomeIntervalDataset(
        bed_file=BED_FILE,
        fasta_file=FASTA_FILE,
        return_augs=True,
        rc_aug=True,
        return_seq_indices=False,
        shift_augs=[-3, 3],
        context_length=524_288,
        filter_df_fn=filter_func,
    )


chromatin_dataset = dict()
for name, dataset in genome_datasets.items():
    chromatin_dataset[name] = ChromatinDataset(
        genome_dataset=dataset,
        bigwig_dir=BIGWIG_DIR,
        clip_soft=32,
        clip_hard=128,
        scale_factor=0.1,  # Scale by about 10x to match the original scale of the data
        power_transform_exponent=1.0,
    )


########################
# MODEL CONFIGURATION
########################

model_config = LongBoiConfig(
    borzoi_model_name="johahi/flashzoi-replicate-0",
    n_labels=chromatin_dataset["train"].n_labels,
    use_autocast=True,
    id2label=chromatin_dataset["train"].id2label,
    label2id=chromatin_dataset["train"].label2id,
    borzoi_kwargs={
        "enable_mouse_head": False,
    }
)
model = LongBoi(config=model_config)
model.init_borzoi_weights()

lora_config = LoraConfig(
    target_modules=r"(?!.*separable\d+)(?!.*_head).*conv_layer|.*Wqkv|transformer\.\d+\.1\.fn\.1|transformer\.\d+\.1\.fn\.4",
    modules_to_save=["chromatin_head"],
    r=8,
    lora_alpha=8,
    lora_dropout=0.05,
    bias="none",
    task_type="FEATURE_EXTRACTION",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
n_trainable_params, n_total_params = model.get_nb_trainable_parameters()

########################
# TRAINING CONFIGURATION
########################

STEPS_EVAL_AND_SAVE = 1000
PROJECT = "LongBoi Chromatin"
GROUP = "LongBoi"
RUN_NAME = (
    f"LongBoi Chromatin Scaled Dataset {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
)
OUTPUT_ROOT = "logs/long_boi_chromatin_scaled_dataset"
params = {
    **model_config.to_dict(),
    "lora_config": lora_config.to_dict(),
    "dataset_params": chromatin_dataset["train"].params,
    "n_trainable_params": n_trainable_params,
    "n_total_params": n_total_params,
    "steps_eval_and_save": STEPS_EVAL_AND_SAVE,
    "project": PROJECT,
    "group": GROUP,
    "run_name": RUN_NAME,
    "output_root": OUTPUT_ROOT,
}

wandb.init(
    project=PROJECT,
    group=GROUP,
    job_type="fine-tune",
    name=RUN_NAME,
    dir=OUTPUT_ROOT,
    reinit="finish_previous",
    config=params,
)


training_args = TrainingArguments(
    bf16_full_eval=False,
    bf16=True,
    dataloader_num_workers=30,
    dataloader_pin_memory=True,
    eval_accumulation_steps=10,
    eval_steps=STEPS_EVAL_AND_SAVE,
    eval_strategy="steps",
    gradient_accumulation_steps=8,
    label_names=["labels"],
    learning_rate=1e-4,
    load_best_model_at_end=True,
    logging_steps=10,
    logging_dir=f"{OUTPUT_ROOT}/logs/{RUN_NAME.replace(' ', '_')}",
    num_train_epochs=5,
    output_dir="checkpoints/long_boi_scaled_dataset",
    per_device_eval_batch_size=1,
    per_device_train_batch_size=2,
    prediction_loss_only=False,
    remove_unused_columns=False,
    report_to="wandb",
    save_steps=STEPS_EVAL_AND_SAVE,
    lr_scheduler_type="cosine",
    save_strategy="steps",
    warmup_ratio=0.05,
    weight_decay=5e-4,
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=chromatin_dataset["train"],
    eval_dataset=chromatin_dataset["val"],
    callbacks=[
        EarlyStoppingCallback(early_stopping_patience=5, early_stopping_threshold=0.5),
        SaveMergedModelCallback(
            save_every=STEPS_EVAL_AND_SAVE,
            save_path_prefix=f"model_weights/LB_Flashzoi_{RUN_NAME.replace(' ', '_')}",
        ),
    ],
    compute_metrics=compute_metrics,
)

##### TRAINING #####
trainer.train()


#### FINALIZATION ####
wandb.finish()
model.merge_and_unload().save_pretrained(
    f"models/{RUN_NAME.replace(' ', '_')}",
)
