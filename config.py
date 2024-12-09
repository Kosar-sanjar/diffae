# config.py

from dataclasses import dataclass, field
from enum import Enum
import os
from model.unet import ScaleAt
from model.latentnet import *
from diffusion.resample import UniformSampler
from diffusion.diffusion import space_timesteps
from typing import Tuple

from torch.utils.data import DataLoader, Subset

from config_base import BaseConfig
from dataset import *
from diffusion import *
from diffusion.base import GenerativeType, LossType, ModelMeanType, ModelVarType, get_named_beta_schedule
from model import *
from choices import *
from multiprocessing import get_context
from dataset_util import *
from torch.utils.data.distributed import DistributedSampler

# Define data paths
data_paths = {
    'ffhqlmdb256': os.path.expanduser('datasets/ffhq256.lmdb'),
    # used for training a classifier
    'celeba': os.path.expanduser('datasets/celeba'),
    # used for training DPM models
    'celebalmdb': os.path.expanduser('datasets/celeba.lmdb'),
    'celebahq': os.path.expanduser('datasets/celebahq256.lmdb'),
    'horse256': os.path.expanduser('datasets/horse256.lmdb'),
    'bedroom256': os.path.expanduser('datasets/bedroom256.lmdb'),
    'celeba_anno': os.path.expanduser('datasets/celeba_anno/list_attr_celeba.txt'),
    'celebahq_anno': os.path.expanduser('datasets/celeba_anno/CelebAMask-HQ-attribute-anno.txt'),
    'celeba_relight': os.path.expanduser('datasets/celeba_hq_light/celeba_light.txt'),
}

@dataclass
class PretrainConfig(BaseConfig):
    name: str
    path: str

# Separate configuration classes for Semantic Encoder and Conditional DDIM
@dataclass
class SemanticEncoderConfig:
    embedding_dim: int
    input_channels: int = 3  # Adjust based on your EEG data's channel count
    hidden_layers: Tuple[int] = (256, 512)  # Example hidden layers
    activation: Activation = Activation.relu  # Example activation

    def make_model(self):
        # Initialize and return the Semantic Encoder model
        return SemanticEncoder(
            embedding_dim=self.embedding_dim,
            input_channels=self.input_channels,
            hidden_layers=self.hidden_layers,
            activation=self.activation.get_act()
        )

@dataclass
class ConditionalDDIMConfig:
    embedding_dim: int
    img_size: int = 64
    output_channels: int = 3  # RGB images
    hidden_layers: Tuple[int] = (256, 512, 1024)  # Example hidden layers
    activation: Activation = Activation.relu  # Example activation

    def make_model(self):
        # Initialize and return the Conditional DDIM model
        return ConditionalDDIM(
            embedding_dim=self.embedding_dim,
            img_size=self.img_size,
            output_channels=self.output_channels,
            hidden_layers=self.hidden_layers,
            activation=self.activation.get_act()
        )

@dataclass
class TrainConfig(BaseConfig):
    # Random seed
    seed: int = 0
    # Training mode
    train_mode: TrainMode = TrainMode.diffusion  # Default mode

    # Paths for Semantic Encoder
    semantic_encoder_lmdb: str = 'datasets/eeg_encoder.lmdb'
    semantic_encoder_checkpoint: str = None  # Path to pre-trained Semantic Encoder (if any)

    # Paths for Conditional DDIM
    conditional_ddim_lmdb: str = 'datasets/eeg_ddim_ffhq.lmdb'
    conditional_ddim_checkpoint: str = None  # Path to pre-trained Conditional DDIM (if any)

    # Limiting datasets to 10 samples for testing
    max_train_samples: int = 10
    max_val_samples: int = 10

    # Existing fields
    train_cond0_prob: float = 0
    train_pred_xstart_detach: bool = True
    train_interpolate_prob: float = 0
    train_interpolate_img: bool = False
    manipulate_mode: ManipulateMode = ManipulateMode.celebahq_all
    manipulate_cls: str = None
    manipulate_shots: int = None
    manipulate_loss: ManipulateLossType = ManipulateLossType.bce
    manipulate_znormalize: bool = False
    manipulate_seed: int = 0
    accum_batches: int = 1
    autoenc_mid_attn: bool = True
    batch_size: int = 16
    batch_size_eval: int = None
    beatgans_gen_type: GenerativeType = GenerativeType.ddim
    beatgans_loss_type: LossType = LossType.mse
    beatgans_model_mean_type: ModelMeanType = ModelMeanType.eps
    beatgans_model_var_type: ModelVarType = ModelVarType.fixed_large
    beatgans_rescale_timesteps: bool = False
    latent_infer_path: str = None
    latent_znormalize: bool = False
    latent_gen_type: GenerativeType = GenerativeType.ddim
    latent_loss_type: LossType = LossType.mse
    latent_model_mean_type: ModelMeanType = ModelMeanType.eps
    latent_model_var_type: ModelVarType = ModelVarType.fixed_large
    latent_rescale_timesteps: bool = False
    latent_T_eval: int = 1_000
    latent_clip_sample: bool = False
    latent_beta_scheduler: str = 'linear'
    beta_scheduler: str = 'linear'
    data_name: str = ''
    data_val_name: str = None
    diffusion_type: str = None
    dropout: float = 0.1
    ema_decay: float = 0.9999
    eval_num_images: int = 5_000
    eval_every_samples: int = 200_000
    eval_ema_every_samples: int = 200_000
    fid_use_torch: bool = True
    fp16: bool = False
    grad_clip: float = 1
    img_size: int = 64
    lr: float = 0.0001
    optimizer: OptimizerType = OptimizerType.adam
    weight_decay: float = 0
    model_conf: ModelConfig = None
    model_name: ModelName = None
    model_type: ModelType = None
    net_attn: Tuple[int] = None
    net_beatgans_attn_head: int = 1
    # Not necessarily the same as the number of style channels
    net_beatgans_embed_channels: int = 512
    net_resblock_updown: bool = True
    net_enc_use_time: bool = False
    net_enc_pool: str = 'adaptivenonzero'
    net_beatgans_gradient_checkpoint: bool = False
    net_beatgans_resnet_two_cond: bool = False
    net_beatgans_resnet_use_zero_module: bool = True
    net_beatgans_resnet_scale_at: ScaleAt = ScaleAt.after_norm
    net_beatgans_resnet_cond_channels: int = None
    net_ch_mult: Tuple[int] = None
    net_ch: int = 64
    net_enc_attn: Tuple[int] = None
    net_enc_k: int = None
    # Number of resblocks for the encoder (half-unet)
    net_enc_num_res_blocks: int = 2
    net_enc_channel_mult: Tuple[int] = None
    net_enc_grad_checkpoint: bool = False
    net_autoenc_stochastic: bool = False
    net_latent_activation: Activation = Activation.silu
    net_latent_channel_mult: Tuple[int] = (1, 2, 4)
    net_latent_condition_bias: float = 0
    net_latent_dropout: float = 0
    net_latent_layers: int = None
    net_latent_net_last_act: Activation = Activation.none
    net_latent_net_type: LatentNetType = LatentNetType.none
    net_latent_num_hid_channels: int = 1024
    net_latent_num_time_layers: int = 2
    net_latent_skip_layers: Tuple[int] = None
    net_latent_time_emb_channels: int = 64
    net_latent_use_norm: bool = False
    net_latent_time_last_act: bool = False
    net_num_res_blocks: int = 2
    # Number of resblocks for the UNET
    net_num_input_res_blocks: int = None
    net_enc_num_cls: int = None
    num_workers: int = 4
    parallel: bool = False
    postfix: str = ''
    sample_size: int = 64
    sample_every_samples: int = 20_000
    save_every_samples: int = 100_000
    style_ch: int = 512
    T_eval: int = 1_000
    T_sampler: str = 'uniform'
    T: int = 1_000
    total_samples: int = 10_000_000
    warmup: int = 0
    pretrain: PretrainConfig = None
    continue_from: PretrainConfig = None
    eval_programs: Tuple[str] = None
    # If present, load the checkpoint from this path instead
    eval_path: str = None
    base_dir: str = 'checkpoints'
    use_cache_dataset: bool = False
    data_cache_dir: str = os.path.expanduser('~/cache')
    work_cache_dir: str = os.path.expanduser('~/mycache')
    # To be overridden
    name: str = ''

    def __post_init__(self):
        self.batch_size_eval = self.batch_size_eval or self.batch_size
        self.data_val_name = self.data_val_name or self.data_name

        # Validation based on train_mode
        if self.train_mode == TrainMode.conditional_ddim:
            if not self.semantic_encoder_checkpoint:
                raise ValueError("semantic_encoder_checkpoint must be provided for Conditional DDIM training.")

    def scale_up_gpus(self, num_gpus, num_nodes=1):
        self.eval_ema_every_samples *= num_gpus * num_nodes
        self.eval_every_samples *= num_gpus * num_nodes
        self.sample_every_samples *= num_gpus * num_nodes
        self.batch_size *= num_gpus * num_nodes
        self.batch_size_eval *= num_gpus * num_nodes
        return self

    @property
    def batch_size_effective(self):
        return self.batch_size * self.accum_batches

    @property
    def fid_cache(self):
        # We try to use the local dirs to reduce the load over network drives
        # Hopefully, this would reduce the disconnection problems with sshfs
        return f'{self.work_cache_dir}/eval_images/{self.data_name}_size{self.img_size}_{self.eval_num_images}'

    @property
    def data_path(self):
        # May use the cache dir
        path = data_paths[self.data_name]
        if self.use_cache_dataset and path is not None:
            path = use_cached_dataset_path(
                path, f'{self.data_cache_dir}/{self.data_name}')
        return path

    @property
    def logdir(self):
        return f'{self.base_dir}/{self.name}'

    @property
    def generate_dir(self):
        # We try to use the local dirs to reduce the load over network drives
        # Hopefully, this would reduce the disconnection problems with sshfs
        return f'{self.work_cache_dir}/gen_images/{self.name}'

    def _make_diffusion_conf(self, T=None):
        if self.diffusion_type == 'beatgans':
            # Can use T < self.T for evaluation
            # Follows the guided-diffusion repo conventions
            # t's are evenly spaced
            if self.beatgans_gen_type == GenerativeType.ddpm:
                section_counts = [T]
            elif self.beatgans_gen_type == GenerativeType.ddim:
                section_counts = f'ddim{T}'
            else:
                raise NotImplementedError()

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
            raise NotImplementedError()

    def _make_latent_diffusion_conf(self, T=None):
        # Can use T < self.T for evaluation
        # Follows the guided-diffusion repo conventions
        # t's are evenly spaced
        if self.latent_gen_type == GenerativeType.ddpm:
            section_counts = [T]
        elif self.latent_gen_type == GenerativeType.ddim:
            section_counts = f'ddim{T}'
        else:
            raise NotImplementedError()

        return SpacedDiffusionBeatGansConfig(
            train_pred_xstart_detach=self.train_pred_xstart_detach,
            gen_type=self.latent_gen_type,
            # Latent's model is always ddpm
            model_type=ModelType.ddpm,
            # Latent shares the beta scheduler and full T
            betas=get_named_beta_schedule(self.latent_beta_scheduler, self.T),
            model_mean_type=self.latent_model_mean_type,
            model_var_type=self.latent_model_var_type,
            loss_type=self.latent_loss_type,
            rescale_timesteps=self.latent_rescale_timesteps,
            use_timesteps=space_timesteps(num_timesteps=self.T,
                                          section_counts=section_counts),
            fp16=self.fp16,
        )

    @property
    def model_out_channels(self):
        return 3

    def make_T_sampler(self):
        if self.T_sampler == 'uniform':
            return UniformSampler(self.T)
        else:
            raise NotImplementedError()

    def make_diffusion_conf(self):
        return self._make_diffusion_conf(self.T)

    def make_eval_diffusion_conf(self):
        return self._make_diffusion_conf(T=self.T_eval)

    def make_latent_diffusion_conf(self):
        return self._make_latent_diffusion_conf(T=self.T)

    def make_latent_eval_diffusion_conf(self):
        # Latent can have different eval T
        return self._make_latent_diffusion_conf(T=self.latent_T_eval)

    def make_dataset(self, path=None, split='train', **kwargs):
        if self.data_name == 'ffhqlmdb256':
            return FFHQlmdb(path=path or self.data_path,
                            image_size=self.img_size,
                            split=split,
                            **kwargs)
        elif self.data_name == 'horse256':
            return Horse_lmdb(path=path or self.data_path,
                              image_size=self.img_size,
                              split=split,
                              **kwargs)
        elif self.data_name == 'bedroom256':
            return Bedroom_lmdb(path=path or self.data_path,
                                image_size=self.img_size,
                                split=split,
                                **kwargs)
        elif self.data_name == 'celebalmdb':
            # Always use d2c crop
            return CelebAlmdb(path=path or self.data_path,
                              image_size=self.img_size,
                              original_resolution=None,
                              crop_d2c=True,
                              split=split,
                              **kwargs)
        else:
            raise NotImplementedError()

    def make_loader(self,
                    dataset,
                    shuffle: bool,
                    num_worker: bool = None,
                    drop_last: bool = True,
                    batch_size: int = None,
                    parallel: bool = False):
        if parallel and distributed.is_initialized():
            # Drop last to make sure that there are no added special indexes
            sampler = DistributedSampler(dataset,
                                         shuffle=shuffle,
                                         drop_last=True)
        else:
            sampler = None
        return DataLoader(
            dataset,
            batch_size=batch_size or self.batch_size,
            sampler=sampler,
            # With sampler, use the sample instead of this option
            shuffle=False if sampler else shuffle,
            num_workers=num_worker or self.num_workers,
            pin_memory=True,
            drop_last=drop_last,
            multiprocessing_context=get_context('fork'),
        )

    def make_semantic_encoder_dataset(self, split='train'):
        """
        Initialize the dataset for Semantic Encoder training.
        Limits to `max_train_samples` for training and `max_val_samples` for validation.
        """
        dataset = EEGEncoderDataset(self.semantic_encoder_lmdb, split=split)
        if split == 'train' and self.max_train_samples:
            # Select the first `max_train_samples` samples for training
            indices = list(range(min(self.max_train_samples, len(dataset))))
            dataset = Subset(dataset, indices)
        elif split == 'val' and self.max_val_samples:
            # Select the first `max_val_samples` samples for validation
            indices = list(range(min(self.max_val_samples, len(dataset))))
            dataset = Subset(dataset, indices)
        return dataset

    def make_conditional_ddim_dataset(self, split='train'):
        """
        Initialize the dataset for Conditional DDIM training.
        Limits to `max_train_samples` for training and `max_val_samples` for validation.
        """
        dataset = ConditionalDDIMDataset(self.conditional_ddim_lmdb, split=split)
        if split == 'train' and self.max_train_samples:
            # Select the first `max_train_samples` samples for training
            indices = list(range(min(self.max_train_samples, len(dataset))))
            dataset = Subset(dataset, indices)
        elif split == 'val' and self.max_val_samples:
            # Select the first `max_val_samples` samples for validation
            indices = list(range(min(self.max_val_samples, len(dataset))))
            dataset = Subset(dataset, indices)
        return dataset

    def make_model_conf(self):
        """
        Create model configuration based on the training mode.
        """
        if self.train_mode == TrainMode.semantic_encoder:
            # Configuration for Semantic Encoder
            return SemanticEncoderConfig(
                embedding_dim=512,
                input_channels=3,  # Adjust based on your EEG data's channel count
                hidden_layers=(256, 512),
                activation=Activation.relu
            )
        elif self.train_mode == TrainMode.conditional_ddim:
            # Configuration for Conditional DDIM
            return ConditionalDDIMConfig(
                embedding_dim=512,
                img_size=self.img_size,
                output_channels=3,
                hidden_layers=(256, 512, 1024),
                activation=Activation.relu
            )
        else:
            # Existing configurations for other modes
            return self._make_diffusion_conf(self.T)
