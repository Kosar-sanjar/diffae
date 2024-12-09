import os
from dataclasses import dataclass, field
from typing import Tuple, Optional, Union, List

import torch
from torch.utils.data import Dataset, DataLoader
from enum import Enum, auto

from model.unet import ScaleAt
from model.latentnet import *
from diffusion.resample import UniformSampler
from diffusion.diffusion import space_timesteps
from diffusion.base import (
    GenerativeType,
    LossType,
    ModelMeanType,
    ModelVarType,
    get_named_beta_schedule
)
from model import *
from choices import *
from dataset_util import *
from torch.utils.data.distributed import DistributedSampler


# Define the available training modes
class TrainMode(Enum):
    encoder = auto()
    latent_diffusion = auto()
    # Add other modes if necessary


# Define Latent Network Types
class LatentNetType(Enum):
    none = auto()
    skip = auto()
    # Add other types if necessary


@dataclass
class PretrainConfig:
    name: str
    path: str


@dataclass
class TrainConfig:
    """
    Configuration class for training and evaluation.
    """
    # -----------------------
    # General Settings
    # -----------------------
    name: str = ''
    seed: int = 0
    base_dir: str = 'checkpoints'
    logdir: str = field(init=False)
    generate_dir: str = field(init=False)
    postfix: str = ''
    
    # -----------------------
    # Data Settings
    # -----------------------
    data_name: str = ''
    data_val_name: Optional[str] = None
    data_path: str = field(init=False)
    data_val_path: Optional[str] = None
    use_cache_dataset: bool = False
    data_cache_dir: str = os.path.expanduser('~/cache')
    work_cache_dir: str = os.path.expanduser('~/mycache')
    eval_num_images: int = 5_000
    num_workers: int = 4
    parallel: bool = False
    sample_size: int = 64
    
    # -----------------------
    # Training Mode
    # -----------------------
    train_mode: TrainMode = TrainMode.encoder
    eval_programs: Tuple[str, ...] = field(default_factory=tuple)
    
    # -----------------------
    # Model Settings
    # -----------------------
    model_name: ModelName = None
    model_type: ModelType = None
    model_conf: Optional['ModelConfig'] = None  # Forward reference
    
    # -----------------------
    # Optimizer Settings
    # -----------------------
    optimizer: OptimizerType = OptimizerType.adam
    lr: float = 0.0001
    weight_decay: float = 0
    warmup: int = 0
    grad_clip: float = 1
    fp16: bool = False
    accum_batches: int = 1
    
    # -----------------------
    # Scheduler Settings
    # -----------------------
    # No specific scheduler settings added here; use defaults or extend as needed
    
    # -----------------------
    # EMA Settings
    # -----------------------
    ema_decay: float = 0.9999
    
    # -----------------------
    # Batch Settings
    # -----------------------
    batch_size: int = 16
    batch_size_eval: Optional[int] = None
    batch_size_effective: int = field(init=False)
    
    # -----------------------
    # Sampling Settings
    # -----------------------
    T: int = 1_000
    T_eval: int = 1_000
    T_sampler: str = 'uniform'
    latent_T_eval: int = 1_000
    latent_gen_type: GenerativeType = GenerativeType.ddim
    latent_beta_scheduler: str = 'linear'
    latent_clip_sample: bool = False
    latent_rescale_timesteps: bool = False
    latent_loss_type: LossType = LossType.mse
    latent_model_mean_type: ModelMeanType = ModelMeanType.eps
    latent_model_var_type: ModelVarType = ModelVarType.fixed_large
    latent_net_type: LatentNetType = LatentNetType.none
    latent_num_hid_channels: int = 1024
    latent_num_time_layers: int = 2
    latent_skip_layers: Optional[Tuple[int, ...]] = None
    latent_time_emb_channels: int = 64
    latent_use_norm: bool = False
    latent_time_last_act: bool = False
    latent_net_last_act: Optional[Union[Activation, None]] = Activation.none
    latent_znormalize: bool = False
    latent_infer_path: Optional[str] = None
    
    # -----------------------
    # Model Architecture Settings
    # -----------------------
    # Encoder-specific settings
    enc_num_res_blocks: int = 2
    enc_channel_mult: Optional[Tuple[int, ...]] = None
    enc_attn: Optional[Tuple[int, ...]] = None
    enc_pool: str = 'adaptivenonzero'
    enc_grad_checkpoint: bool = False
    enc_use_time: bool = False
    enc_num_cls: Optional[int] = None
    
    # DDIM-specific settings
    beatgans_gen_type: GenerativeType = GenerativeType.ddim
    beatgans_loss_type: LossType = LossType.mse
    beatgans_model_mean_type: ModelMeanType = ModelMeanType.eps
    beatgans_model_var_type: ModelVarType = ModelVarType.fixed_large
    beatgans_rescale_timesteps: bool = False
    beatgans_embed_channels: int = 512
    beatgans_attn_head: int = 1
    beatgans_gradient_checkpoint: bool = False
    beatgans_resnet_two_cond: bool = False
    beatgans_resnet_use_zero_module: bool = True
    beatgans_resnet_scale_at: ScaleAt = ScaleAt.after_norm
    beatgans_resnet_cond_channels: Optional[int] = None
    
    # -----------------------
    # Latent Network Settings
    # -----------------------
    latent_activation: Activation = Activation.silu
    latent_channel_mult: Tuple[int, ...] = (1, 2, 4)
    latent_condition_bias: float = 0
    latent_dropout: float = 0
    
    # -----------------------
    # Other Settings
    # -----------------------
    dropout: float = 0.1
    # Add other necessary fields as required
    
    # -----------------------
    # Pretrain and Continue
    # -----------------------
    pretrain: Optional[PretrainConfig] = None
    continue_from: Optional[PretrainConfig] = None
    
    def __post_init__(self):
        # Initialize derived fields
        self.batch_size_eval = self.batch_size_eval or self.batch_size
        self.batch_size_effective = self.batch_size * self.accum_batches
        self.data_val_name = self.data_val_name or self.data_name
        self.logdir = f'{self.base_dir}/{self.name}'
        self.generate_dir = f'{self.work_cache_dir}/gen_images/{self.name}'
        self.data_val_path = data_paths.get(self.data_val_name, None)
        self.data_path = data_paths.get(self.data_name, None)
        if self.use_cache_dataset and self.data_path is not None:
            self.data_path = use_cached_dataset_path(
                self.data_path, f'{self.data_cache_dir}/{self.data_name}')
    
    def scale_up_gpus(self, num_gpus, num_nodes=1):
        """
        Scale batch sizes and sampling frequencies based on the number of GPUs and nodes.
        """
        self.eval_ema_every_samples *= num_gpus * num_nodes
        self.eval_every_samples *= num_gpus * num_nodes
        self.sample_every_samples *= num_gpus * num_nodes
        self.batch_size *= num_gpus * num_nodes
        self.batch_size_eval *= num_gpus * num_nodes
        return self
    
    def make_T_sampler(self):
        """
        Create a timestep sampler based on the configuration.
        """
        if self.T_sampler == 'uniform':
            return UniformSampler(self.T)
        else:
            raise NotImplementedError(f"Sampler type '{self.T_sampler}' is not supported.")
    
    def _make_diffusion_conf(self, T=None):
        """
        Create a diffusion configuration for BeatGANs.
        """
        if self.diffusion_type == 'beatgans':
            if self.beatgans_gen_type == GenerativeType.ddpm:
                section_counts = [T]
            elif self.beatgans_gen_type == GenerativeType.ddim:
                section_counts = f'ddim{T}'
            else:
                raise NotImplementedError(f"GenerativeType '{self.beatgans_gen_type}' is not supported.")
    
            return SpacedDiffusionBeatGansConfig(
                gen_type=self.beatgans_gen_type,
                model_type=self.model_type,
                betas=get_named_beta_schedule(self.beta_scheduler, self.T),
                model_mean_type=self.beatgans_model_mean_type,
                model_var_type=self.beatgans_model_var_type,
                loss_type=self.beatgans_loss_type,
                rescale_timesteps=self.beatgans_rescale_timesteps,
                use_timesteps=space_timesteps(num_timesteps=self.T,
                                              section_counts=section_counts),
                fp16=self.fp16,
            )
        else:
            raise NotImplementedError(f"Diffusion type '{self.diffusion_type}' is not supported.")
    
    def _make_latent_diffusion_conf(self, T=None):
        """
        Create a diffusion configuration for latent diffusion.
        """
        if self.latent_gen_type == GenerativeType.ddpm:
            section_counts = [T]
        elif self.latent_gen_type == GenerativeType.ddim:
            section_counts = f'ddim{T}'
        else:
            raise NotImplementedError(f"GenerativeType '{self.latent_gen_type}' is not supported.")
    
        return SpacedDiffusionBeatGansConfig(
            train_pred_xstart_detach=self.train_pred_xstart_detach,
            gen_type=self.latent_gen_type,
            model_type=ModelType.ddpm,  # Latent models are always ddpm
            betas=get_named_beta_schedule(self.latent_beta_scheduler, self.T),
            model_mean_type=self.latent_model_mean_type,
            model_var_type=self.latent_model_var_type,
            loss_type=self.latent_loss_type,
            rescale_timesteps=self.latent_rescale_timesteps,
            use_timesteps=space_timesteps(num_timesteps=self.T,
                                          section_counts=section_counts),
            fp16=self.fp16,
        )
    
    def make_diffusion_conf(self):
        """
        Public method to create a diffusion configuration based on train mode.
        """
        if self.train_mode == TrainMode.encoder:
            return self._make_diffusion_conf(self.T)
        elif self.train_mode == TrainMode.latent_diffusion:
            return self._make_latent_diffusion_conf(self.T)
        else:
            raise NotImplementedError(f"TrainMode '{self.train_mode}' is not supported.")
    
    def make_eval_diffusion_conf(self):
        """
        Create an evaluation diffusion configuration.
        """
        return self._make_diffusion_conf(T=self.T_eval)
    
    def make_latent_diffusion_conf(self):
        """
        Create a latent diffusion configuration.
        """
        return self._make_latent_diffusion_conf(T=self.T)
    
    def make_latent_eval_diffusion_conf(self):
        """
        Create a latent diffusion evaluation configuration.
        """
        return self._make_latent_diffusion_conf(T=self.latent_T_eval)
    
    def make_dataset(self, path: Optional[str] = None, **kwargs) -> Dataset:
        """
        Create a dataset based on the data name and training mode.
        """
        if self.train_mode == TrainMode.encoder:
            # Encoder training: only EEG data
            return EEGDataset(path=path or self.data_path, **kwargs)
        elif self.train_mode == TrainMode.latent_diffusion:
            # DDIM training: EEG and Image data
            return ConditionalEEGImageDataset(path=path or self.data_path, **kwargs)
        else:
            raise NotImplementedError(f"TrainMode '{self.train_mode}' is not supported.")
    
    def make_validation_dataset(self, path: Optional[str] = None, **kwargs) -> Dataset:
        """
        Create a validation dataset. Can be the same as training or separate.
        """
        if self.data_val_name:
            val_path = path or self.data_val_path
            if self.train_mode == TrainMode.encoder:
                return EEGDataset(path=val_path, **kwargs)
            elif self.train_mode == TrainMode.latent_diffusion:
                return ConditionalEEGImageDataset(path=val_path, **kwargs)
            else:
                raise NotImplementedError(f"TrainMode '{self.train_mode}' is not supported.")
        else:
            return self.make_dataset(path=path, **kwargs)
    
    def make_loader(self,
                   dataset: Dataset,
                   shuffle: bool,
                   num_worker: Optional[int] = None,
                   drop_last: bool = True,
                   batch_size: Optional[int] = None,
                   parallel: bool = False) -> DataLoader:
        """
        Create a DataLoader with optional distributed sampling.
        """
        if parallel and torch.distributed.is_initialized():
            sampler = DistributedSampler(dataset,
                                         shuffle=shuffle,
                                         drop_last=True)
        else:
            sampler = None
    
        return DataLoader(
            dataset,
            batch_size=batch_size or self.batch_size,
            sampler=sampler,
            shuffle=False if sampler else shuffle,
            num_workers=num_worker or self.num_workers,
            pin_memory=True,
            drop_last=drop_last,
            multiprocessing_context='fork',  # Adjust if necessary
        )
    
    def make_model_conf(self) -> 'ModelConfig':
        """
        Create a model configuration based on the model name.
        """
        if self.model_name == ModelName.beatgans_ddpm:
            self.model_type = ModelType.ddpm
            self.model_conf = BeatGANsUNetConfig(
                attention_resolutions=self.enc_attn,
                channel_mult=self.enc_channel_mult,
                conv_resample=True,
                dims=2,
                dropout=self.dropout,
                embed_channels=self.beatgans_embed_channels,
                image_size=self.img_size,
                in_channels=3,
                model_channels=self.net_ch,
                num_classes=None,
                num_head_channels=-1,
                num_heads_upsample=-1,
                num_heads=self.beatgans_attn_head,
                num_res_blocks=self.net_num_res_blocks,
                num_input_res_blocks=self.net_num_input_res_blocks,
                out_channels=self.model_out_channels,
                resblock_updown=self.net_resblock_updown,
                use_checkpoint=self.beatgans_gradient_checkpoint,
                use_new_attention_order=False,
                resnet_two_cond=self.beatgans_resnet_two_cond,
                resnet_use_zero_module=self.beatgans_resnet_use_zero_module,
            )
        elif self.model_name == ModelName.beatgans_autoenc:
            self.model_type = ModelType.autoencoder
            if self.latent_net_type == LatentNetType.none:
                latent_net_conf = None
            elif self.latent_net_type == LatentNetType.skip:
                latent_net_conf = MLPSkipNetConfig(
                    num_channels=self.style_ch,
                    skip_layers=self.latent_skip_layers,
                    num_hid_channels=self.latent_num_hid_channels,
                    num_layers=self.latent_layers,
                    num_time_emb_channels=self.latent_time_emb_channels,
                    activation=self.latent_activation,
                    use_norm=self.latent_use_norm,
                    condition_bias=self.latent_condition_bias,
                    dropout=self.latent_dropout,
                    last_act=self.latent_net_last_act,
                    num_time_layers=self.latent_num_time_layers,
                    time_last_act=self.latent_time_last_act,
                )
            else:
                raise NotImplementedError(f"LatentNetType '{self.latent_net_type}' is not supported.")
    
            self.model_conf = BeatGANsAutoencConfig(
                attention_resolutions=self.enc_attn,
                channel_mult=self.enc_channel_mult,
                conv_resample=True,
                dims=2,
                dropout=self.dropout,
                embed_channels=self.beatgans_embed_channels,
                enc_out_channels=self.style_ch,
                enc_pool=self.enc_pool,
                enc_num_res_block=self.enc_num_res_blocks,
                enc_channel_mult=self.enc_channel_mult,
                enc_grad_checkpoint=self.enc_grad_checkpoint,
                enc_attn_resolutions=self.enc_attn,
                image_size=self.img_size,
                in_channels=3,
                model_channels=self.net_ch,
                num_classes=None,
                num_head_channels=-1,
                num_heads_upsample=-1,
                num_heads=self.beatgans_attn_head,
                num_res_blocks=self.net_num_res_blocks,
                num_input_res_blocks=self.net_num_input_res_blocks,
                out_channels=self.model_out_channels,
                resblock_updown=self.net_resblock_updown,
                use_checkpoint=self.beatgans_gradient_checkpoint,
                use_new_attention_order=False,
                resnet_two_cond=self.beatgans_resnet_two_cond,
                resnet_use_zero_module=self.beatgans_resnet_use_zero_module,
                latent_net_conf=latent_net_conf,
                resnet_cond_channels=self.beatgans_resnet_cond_channels,
            )
        else:
            raise NotImplementedError(f"ModelName '{self.model_name}' is not supported.")
    
        return self.model_conf
    
    def make_validation_loader(self) -> DataLoader:
        """
        Create a DataLoader for validation.
        """
        val_dataset = self.make_validation_dataset()
        return self.make_loader(
            dataset=val_dataset,
            shuffle=False,
            drop_last=False,
            batch_size=self.batch_size_eval,
            parallel=self.parallel
        )


# ---------------------------
# Additional Helper Classes
# ---------------------------

@dataclass
class ModelConfig:
    """
    Base class for model configurations.
    """
    pass  # To be extended by specific model configurations


@dataclass
class BeatGANsUNetConfig(ModelConfig):
    """
    Configuration for BeatGANs UNet model.
    """
    attention_resolutions: Tuple[int, ...]
    channel_mult: Tuple[int, ...]
    conv_resample: bool
    dims: int
    dropout: float
    embed_channels: int
    image_size: int
    in_channels: int
    model_channels: int
    num_classes: Optional[int]
    num_head_channels: int
    num_heads_upsample: int
    num_heads: int
    num_res_blocks: int
    num_input_res_blocks: Optional[int]
    out_channels: int
    resblock_updown: bool
    use_checkpoint: bool
    use_new_attention_order: bool
    resnet_two_cond: bool
    resnet_use_zero_module: bool
    resnet_cond_channels: Optional[int] = None


@dataclass
class BeatGANsAutoencConfig(ModelConfig):
    """
    Configuration for BeatGANs Autoencoder model.
    """
    attention_resolutions: Tuple[int, ...]
    channel_mult: Tuple[int, ...]
    conv_resample: bool
    dims: int
    dropout: float
    embed_channels: int
    enc_out_channels: int
    enc_pool: str
    enc_num_res_block: int
    enc_channel_mult: Tuple[int, ...]
    enc_grad_checkpoint: bool
    enc_attn_resolutions: Tuple[int, ...]
    image_size: int
    in_channels: int
    model_channels: int
    num_classes: Optional[int]
    num_head_channels: int
    num_heads_upsample: int
    num_heads: int
    num_res_blocks: int
    num_input_res_blocks: Optional[int]
    out_channels: int
    resblock_updown: bool
    use_checkpoint: bool
    use_new_attention_order: bool
    resnet_two_cond: bool
    resnet_use_zero_module: bool
    latent_net_conf: Optional['MLPSkipNetConfig'] = None
    resnet_cond_channels: Optional[int] = None


@dataclass
class MLPSkipNetConfig:
    """
    Configuration for MLP Skip Network.
    """
    num_channels: int
    skip_layers: Optional[Tuple[int, ...]]
    num_hid_channels: int
    num_layers: int
    num_time_emb_channels: int
    activation: Activation
    use_norm: bool
    condition_bias: float
    dropout: float
    last_act: Optional[Activation]
    num_time_layers: int
    time_last_act: bool


# ---------------------------
# Dataset Classes
# ---------------------------

@dataclass
class EEGDataset(Dataset):
    """
    Dataset class for EEG data (Semantic Encoder Training).
    """
    path: str
    transform: Optional[callable] = None  # Add transforms if necessary
    
    def __len__(self):
        # Implement length based on LMDB contents or data structure
        return 0  # Placeholder
    
    def __getitem__(self, idx):
        # Implement data retrieval from LMDB or other storage
        return torch.tensor([])  # Placeholder


@dataclass
class ConditionalEEGImageDataset(Dataset):
    """
    Dataset class for Conditional DDIM Training (EEG + Image).
    """
    path: str
    transform: Optional[callable] = None  # Add transforms if necessary
    
    def __len__(self):
        # Implement length based on LMDB contents or data structure
        return 0  # Placeholder
    
    def __getitem__(self, idx):
        # Implement data retrieval from LMDB or other storage
        eeg = torch.tensor([])  # Placeholder EEG data
        image = torch.tensor([])  # Placeholder Image data
        return eeg, image


# ---------------------------
# Data Paths
# ---------------------------

data_paths = {
    'ffhqlmdb256': os.path.expanduser('datasets/ffhq256.lmdb'),
    'eeg_encoder': os.path.expanduser('datasets/eeg_encoder.lmdb'),  # Added for encoder training
    'eeg_ddim_ffhq': os.path.expanduser('datasets/eeg_ddim_ffhq.lmdb'),  # Added for DDIM training
    'celeba': os.path.expanduser('datasets/celeba'),
    'celebalmdb': os.path.expanduser('datasets/celeba.lmdb'),
    'celebahq': os.path.expanduser('datasets/celebahq256.lmdb'),
    'horse256': os.path.expanduser('datasets/horse256.lmdb'),
    'bedroom256': os.path.expanduser('datasets/bedroom256.lmdb'),
    'celeba_anno': os.path.expanduser('datasets/celeba_anno/list_attr_celeba.txt'),
    'celebahq_anno': os.path.expanduser('datasets/celeba_anno/CelebAMask-HQ-attribute-anno.txt'),
    'celeba_relight': os.path.expanduser('datasets/celeba_hq_light/celeba_light.txt'),
}

def use_cached_dataset_path(original_path: str, cache_dir: str) -> str:
    """
    Function to determine the cached dataset path.
    """
    # Implement caching logic if necessary
    return original_path  # Placeholder


# ---------------------------
# Activation Enum
# ---------------------------

class Activation(Enum):
    relu = auto()
    silu = auto()
    leaky_relu = auto()
    tanh = auto()
    sigmoid = auto()
    none = auto()
    # Add other activations as needed


# ---------------------------
# Additional Helper Functions
# ---------------------------

def make_transform(img_size: int, flip_prob: float = 0.5, crop_d2c: bool = False):
    """
    Create a transformation pipeline for images.
    """
    import torchvision.transforms as transforms
    transform_list = []
    if crop_d2c:
        # Implement specific cropping if required
        pass  # Placeholder
    transform_list.append(transforms.Resize(img_size))
    transform_list.append(transforms.CenterCrop(img_size))
    transform_list.append(transforms.ToTensor())
    if flip_prob > 0:
        transform_list.append(transforms.RandomHorizontalFlip(p=flip_prob))
    return transforms.Compose(transform_list)


# ---------------------------
# Utility Classes and Enums
# ---------------------------

class ModelName(Enum):
    beatgans_ddpm = auto()
    beatgans_autoenc = auto()
    # Add other model names as necessary


class ModelType(Enum):
    ddpm = auto()
    autoencoder = auto()
    # Add other model types as necessary


# ---------------------------
# Beta Schedule Function
# ---------------------------

def get_named_beta_schedule(schedule_name: str, num_timesteps: int) -> List[float]:
    """
    Return a list of betas for a given named beta schedule.
    """
    if schedule_name == "linear":
        return list(torch.linspace(0.0001, 0.02, num_timesteps).numpy())
    elif schedule_name == "cosine":
        # Implement cosine schedule if necessary
        raise NotImplementedError("Cosine schedule is not implemented.")
    else:
        raise NotImplementedError(f"Beta schedule '{schedule_name}' is not supported.")
