# Test configuration and fixtures

import pytest
import torch


@pytest.fixture
def sample_data():
    """Fixture providing sample data for testing."""
    return {
        "input_ids": torch.randint(0, 1000, (2, 10)),
        "attention_mask": torch.ones(2, 10),
        "labels": torch.randint(0, 2, (2,)),
    }


@pytest.fixture
def temp_model_dir(tmp_path):
    """Fixture providing a temporary directory for model files."""
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    return model_dir


@pytest.fixture
def config():
    """Fixture providing basic configuration for testing."""
    return {
        "model_name": "distilbert-base-uncased",
        "num_classes": 2,
        "batch_size": 4,
        "learning_rate": 2e-5,
        "max_epochs": 1,
    }
