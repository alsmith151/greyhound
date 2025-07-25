import torch
from borzoi_pytorch import Borzoi
from borzoi_pytorch.config_borzoi import BorzoiConfig
from transformers import PretrainedConfig, PreTrainedModel

from .losses import poisson_multinomial_combined_loss


class GreyhoundConfig(PretrainedConfig):
    model_type = "Greyhound"

    def __init__(self, borzoi_model_name: str = None, n_labels=1, **kwargs):
        """
        Args:
            borzoi_model_name (str): Name of the Borzoi model to use. If None, defaults to "johahi/borzoi-replicate-0".
            n_labels (int): Number of labels for the chromatin prediction task.
            id2label (dict, optional): Mapping from label IDs to label names.
            label2id (dict, optional): Mapping from label names to label IDs.
        """
        super().__init__(**kwargs)
        if borzoi_model_name is None:
            borzoi_model_name = "johahi/borzoi-replicate-0"
        self.borzoi_model_name = borzoi_model_name
        self.n_labels = n_labels
        self.use_autocast = kwargs.get("use_autocast", True)
        self.borzoi_kwargs = kwargs.get("borzoi_kwargs", {})
        self.id2label = kwargs.get("id2label", None)
        self.label2id = kwargs.get("label2id", None)
        if not self.id2label and not self.label2id:
            self.id2label = {i: str(i) for i in range(n_labels)}
            self.label2id = {v: k for k, v in self.id2label.items()}


class Greyhound(PreTrainedModel):
    config_class = GreyhoundConfig

    def __init__(self, config: GreyhoundConfig):
        """
        Initialize the Greyhound model with the given configuration.
        Args:
            config (GreyhoundConfig): Configuration object containing model parameters.
        """
        super().__init__(config)
        self._borzoi_config = BorzoiConfig.from_pretrained(
            config.borzoi_model_name,
        )
        self.borzoi = Borzoi(config=self._borzoi_config)
        self.chromatin_head = torch.nn.Conv1d(
            in_channels=1920, out_channels=config.n_labels, kernel_size=1
        )
        self.final_softplus = torch.nn.Softplus()

    def init_borzoi_weights(self):
        """
        Load weights for the Borzoi model from a specified path. Only needs to be called once
        to initialize the Borzoi weights.

        Pre-trained models do not need to be re-initialized after the first time.
        This method loads the weights from the Borzoi model specified in the configuration.

        Args:
            path (Union[str, Path]): Path to the directory containing Borzoi weights.
        """
        pretrained = Borzoi.from_pretrained(self.config.borzoi_model_name)
        self.borzoi.load_state_dict(pretrained.state_dict(), strict=True)
        print(
            f"Loaded Borzoi weights from {self.config.borzoi_model_name} into Greyhound model."
        )

    def forward(self, input_ids=None, labels=None, **kwargs):
        """
        Forward pass for the Greyhound model.
        Args:
            input_ids (torch.Tensor): Input tensor of shape (batch_size, sequence_length, 4).
            attention_mask (torch.Tensor, optional): Attention mask tensor.
            labels (torch.Tensor, optional): Labels for training.
            **kwargs: Additional keyword arguments.
        Returns:
            torch.Tensor: Logits for the model output.
            Optional[torch.Tensor]: Loss if labels are provided.
        """

        with torch.amp.autocast("cuda", enabled=self.config.use_autocast):
            x = input_ids
            x = self.borzoi.get_embs_after_crop(x)
            x = self.borzoi.final_joined_convs(x)

        x = self.chromatin_head(x)
        logits = self.final_softplus(x)

        loss = None
        if labels is not None:
            # Train with a poisson multinomial loss
            if self.config.n_labels == 1:
                loss = poisson_multinomial_combined_loss(
                    logits.squeeze(1), labels.squeeze(1)
                )
            else:
                loss = poisson_multinomial_combined_loss(logits, labels)
        return logits if loss is None else (loss, logits)
