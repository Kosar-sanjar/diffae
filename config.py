from enum import Enum, auto
from dataclasses import dataclass, field
from choices import TrainMode, ModelType  # Import TrainMode and ModelType


@dataclass
class PretrainConfig:
    name: str
    path: str
    latent_infer_path: str


@dataclass
class TrainConfig:
    # Basic training parameters
    batch_size: int = 32
    lr: float = 1e-4
    epochs: int = 100  # Example default
    train_mode: TrainMode = TrainMode.diffusion  # Default mode
    model_type: ModelType = ModelType.ddpm  # Default model type
    eval_programs: list = field(default_factory=list)
    data_name: str = 'ffhqlmdb256'
    model_name: str = 'beatgans_ddpm'
    # Add other necessary configuration fields

    # Example method to scale batch size based on GPUs
    def scale_up_gpus(self, num_gpus: int):
        self.batch_size *= num_gpus
        # Adjust other parameters as needed

    def make_model_conf(self):
        # Adjust model configuration based on train_mode and model_type
        if self.model_type == ModelType.autoencoder:
            # Configure encoder-specific settings
            self.model_name = 'beatgans_autoenc'  # Example
            # Set other encoder-specific parameters
        elif self.model_type == ModelType.ddpm:
            # Configure DDPM-specific settings
            self.model_name = 'beatgans_ddpm'
            # Set other DDPM-specific parameters
        # Continue as needed for other model types
