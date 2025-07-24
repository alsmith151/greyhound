"""Greyhound ML Package.

A modern Python package for machine learning model training with Transformers.
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from greyhound.models import TransformerClassifier, CustomModel
from greyhound.data import DataModule, HuggingFaceDataModule
from greyhound.trainers import TransformersTrainer

__all__ = ["TransformerClassifier", "CustomModel", "DataModule", "HuggingFaceDataModule", "TransformersTrainer"]
