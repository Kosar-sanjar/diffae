# templates.py

from experiment import *
from config import TrainMode  # Ensure TrainMode is imported
from choices import *
from config import PretrainConfig  # Ensure PretrainConfig is imported


def ddpm(train_mode: TrainMode = TrainMode.latent_diffusion):
    """
    Base configuration for all DDPM-based models.
    
    Args:
        train_mode (TrainMode, optional): Specifies the training mode.
                                          Defaults to TrainMode.latent_diffusion.
    
    Returns:
        TrainConfig: Configured training parameters.
    """
    conf = TrainConfig()
    conf.train_mode = train_mode
    conf.batch_size = 32
    conf.beatgans_gen_type = GenerativeType.ddim
    conf.beta_scheduler = 'linear'
    conf.data_name = 'ffhqlmdb256' if train_mode == TrainMode.latent_diffusion else 'eeg_encoder'
    conf.diffusion_type = 'beatgans'
    conf.eval_ema_every_samples = 200_000
    conf.eval_every_samples = 200_000
    conf.fp16 = True
    conf.lr = 1e-4
    conf.model_name = ModelName.beatgans_ddpm
    conf.net_attn = (16, )
    conf.net_beatgans_attn_head = 1
    conf.net_beatgans_embed_channels = 512
    conf.net_ch_mult = (1, 2, 4, 8)
    conf.net_ch = 64
    conf.sample_size = 32
    conf.T_eval = 20
    conf.T = 1000
    conf.make_model_conf()
    return conf


def autoenc_base(train_mode: TrainMode = TrainMode.encoder):
    """
    Base configuration for all Diff-AE models.
    
    Args:
        train_mode (TrainMode, optional): Specifies the training mode.
                                          Defaults to TrainMode.encoder.
    
    Returns:
        TrainConfig: Configured training parameters.
    """
    conf = TrainConfig()
    conf.train_mode = train_mode
    conf.batch_size = 32
    conf.beatgans_gen_type = GenerativeType.ddim
    conf.beta_scheduler = 'linear'
    conf.data_name = 'ffhqlmdb256' if train_mode == TrainMode.latent_diffusion else 'eeg_encoder'
    conf.diffusion_type = 'beatgans'
    conf.eval_ema_every_samples = 200_000
    conf.eval_every_samples = 200_000
    conf.fp16 = True
    conf.lr = 1e-4
    conf.model_name = ModelName.beatgans_autoenc
    conf.net_attn = (16, )
    conf.net_beatgans_attn_head = 1
    conf.net_beatgans_embed_channels = 512
    conf.net_beatgans_resnet_two_cond = True
    conf.net_ch_mult = (1, 2, 4, 8)
    conf.net_ch = 64
    conf.net_enc_channel_mult = (1, 2, 4, 8, 8)
    conf.net_enc_pool = 'adaptivenonzero'
    conf.sample_size = 32
    conf.T_eval = 20
    conf.T = 1000
    conf.make_model_conf()
    return conf


def ffhq64_ddpm():
    """
    Configuration for FFHQ dataset with image size 64 and DDPM training.
    """
    conf = ddpm(train_mode=TrainMode.latent_diffusion)
    conf.data_name = 'ffhqlmdb256'
    conf.warmup = 0
    conf.total_samples = 72_000_000
    conf.scale_up_gpus(num_gpus=4)
    return conf


def ffhq64_autoenc():
    """
    Configuration for FFHQ dataset with image size 64 and Autoencoder training.
    """
    conf = autoenc_base(train_mode=TrainMode.latent_diffusion)
    conf.data_name = 'ffhqlmdb256'
    conf.warmup = 0
    conf.total_samples = 72_000_000
    conf.net_ch_mult = (1, 2, 4, 8)
    conf.net_enc_channel_mult = (1, 2, 4, 8, 8)
    conf.eval_every_samples = 1_000_000
    conf.eval_ema_every_samples = 1_000_000
    conf.scale_up_gpus(num_gpus=4)
    conf.make_model_conf()
    return conf


def celeba64d2c_ddpm():
    """
    Configuration for CelebA dataset with D2C cropping and DDPM training.
    """
    conf = ffhq128_ddpm()
    conf.data_name = 'celebalmdb'
    conf.eval_every_samples = 10_000_000
    conf.eval_ema_every_samples = 10_000_000
    conf.total_samples = 72_000_000
    conf.name = 'celeba64d2c_ddpm'
    return conf


def celeba64d2c_autoenc():
    """
    Configuration for CelebA dataset with D2C cropping and Autoencoder training.
    """
    conf = ffhq64_autoenc()
    conf.data_name = 'celebalmdb'
    conf.eval_every_samples = 10_000_000
    conf.eval_ema_every_samples = 10_000_000
    conf.total_samples = 72_000_000
    conf.name = 'celeba64d2c_autoenc'
    return conf


def ffhq128_ddpm():
    """
    Configuration for FFHQ dataset with image size 128 and DDPM training.
    """
    conf = ddpm(train_mode=TrainMode.latent_diffusion)
    conf.data_name = 'ffhqlmdb256'
    conf.warmup = 0
    conf.total_samples = 48_000_000
    conf.img_size = 128
    conf.net_ch = 128
    # Channels:
    # 3 => 128 * 1 => 128 * 1 => 128 * 2 => 128 * 3 => 128 * 4
    # Sizes:
    # 128 => 128 => 64 => 32 => 16 => 8
    conf.net_ch_mult = (1, 1, 2, 3, 4)
    conf.eval_every_samples = 10_000_000
    conf.eval_ema_every_samples = 10_000_000
    conf.scale_up_gpus(num_gpus=4)
    conf.make_model_conf()
    return conf


def ffhq128_autoenc_base():
    """
    Base configuration for FFHQ dataset with image size 128 and Autoencoder training.
    """
    conf = autoenc_base(train_mode=TrainMode.latent_diffusion)
    conf.data_name = 'ffhqlmdb256'
    conf.scale_up_gpus(num_gpus=4)
    conf.img_size = 128
    conf.net_ch = 128
    # Final resolution = 8x8
    conf.net_ch_mult = (1, 1, 2, 3, 4)
    # Final resolution = 4x4
    conf.net_enc_channel_mult = (1, 1, 2, 3, 4, 4)
    conf.eval_ema_every_samples = 10_000_000
    conf.eval_every_samples = 10_000_000
    conf.make_model_conf()
    return conf


def ffhq256_autoenc():
    """
    Configuration for FFHQ dataset with image size 256 and Autoencoder training.
    """
    conf = ffhq128_autoenc_base()
    conf.img_size = 256
    conf.net_ch = 128
    conf.net_ch_mult = (1, 1, 2, 2, 4, 4)
    conf.net_enc_channel_mult = (1, 1, 2, 2, 4, 4, 4)
    conf.eval_every_samples = 10_000_000
    conf.eval_ema_every_samples = 10_000_000
    conf.total_samples = 200_000_000
    conf.batch_size = 64
    conf.make_model_conf()
    conf.name = 'ffhq256_autoenc'
    return conf


def ffhq256_autoenc_eco():
    """
    Configuration for FFHQ dataset with image size 256, Autoencoder training, and ECO settings.
    """
    conf = ffhq128_autoenc_base()
    conf.img_size = 256
    conf.net_ch = 128
    conf.net_ch_mult = (1, 1, 2, 2, 4, 4)
    conf.net_enc_channel_mult = (1, 1, 2, 2, 4, 4, 4)
    conf.eval_every_samples = 10_000_000
    conf.eval_ema_every_samples = 10_000_000
    conf.total_samples = 200_000_000
    conf.batch_size = 64
    conf.make_model_conf()
    conf.name = 'ffhq256_autoenc_eco'
    return conf


def ffhq128_ddpm_72M():
    """
    Configuration for FFHQ dataset with image size 128, DDPM training, and 72M total samples.
    """
    conf = ffhq128_ddpm()
    conf.total_samples = 72_000_000
    conf.name = 'ffhq128_ddpm_72M'
    return conf


def ffhq128_autoenc_72M():
    """
    Configuration for FFHQ dataset with image size 128, Autoencoder training, and 72M total samples.
    """
    conf = ffhq128_autoenc_base()
    conf.total_samples = 72_000_000
    conf.name = 'ffhq128_autoenc_72M'
    return conf


def ffhq128_ddpm_130M():
    """
    Configuration for FFHQ dataset with image size 128, DDPM training, and 130M total samples.
    """
    conf = ffhq128_ddpm()
    conf.total_samples = 130_000_000
    conf.eval_ema_every_samples = 10_000_000
    conf.eval_every_samples = 10_000_000
    conf.name = 'ffhq128_ddpm_130M'
    return conf


def horse128_ddpm():
    """
    Configuration for Horse dataset with image size 128 and DDPM training.
    """
    conf = ffhq128_ddpm()
    conf.data_name = 'horse256'
    conf.total_samples = 130_000_000
    conf.eval_ema_every_samples = 10_000_000
    conf.eval_every_samples = 10_000_000
    conf.name = 'horse128_ddpm'
    return conf


def horse128_autoenc():
    """
    Configuration for Horse dataset with image size 128 and Autoencoder training.
    """
    conf = ffhq128_autoenc_base()
    conf.data_name = 'horse256'
    conf.total_samples = 130_000_000
    conf.eval_ema_every_samples = 10_000_000
    conf.eval_every_samples = 10_000_000
    conf.name = 'horse128_autoenc'
    return conf


def bedroom128_ddpm():
    """
    Configuration for Bedroom dataset with image size 128 and DDPM training.
    """
    conf = ffhq128_ddpm()
    conf.data_name = 'bedroom256'
    conf.eval_ema_every_samples = 10_000_000
    conf.eval_every_samples = 10_000_000
    conf.total_samples = 120_000_000
    conf.name = 'bedroom128_ddpm'
    return conf


def bedroom128_autoenc():
    """
    Configuration for Bedroom dataset with image size 128 and Autoencoder training.
    """
    conf = ffhq128_autoenc_base()
    conf.data_name = 'bedroom256'
    conf.eval_ema_every_samples = 10_000_000
    conf.eval_every_samples = 10_000_000
    conf.total_samples = 120_000_000
    conf.name = 'bedroom128_autoenc'
    return conf


def pretrain_celeba64d2c_72M():
    """
    Pretraining configuration for CelebA dataset with D2C cropping and 72M total samples.
    """
    conf = celeba64d2c_autoenc()
    conf.pretrain = PretrainConfig(
        name='72M',
        path=f'checkpoints/{conf.name}/last.ckpt',
    )
    conf.latent_infer_path = f'checkpoints/{conf.name}/latent.pkl'
    return conf


def pretrain_ffhq128_autoenc72M():
    """
    Pretraining configuration for FFHQ dataset with image size 128, Autoencoder training, and 72M total samples.
    """
    conf = ffhq128_autoenc_72M()
    conf.pretrain = PretrainConfig(
        name='72M',
        path=f'checkpoints/{conf.name}/last.ckpt',
    )
    conf.latent_infer_path = f'checkpoints/{conf.name}/latent.pkl'
    return conf

def ffhq128_autoenc_130M():
    """
    Configuration for FFHQ dataset with image size 128 and Autoencoder training.
    """
    conf = TrainConfig()
    conf.batch_size = 32
    conf.beatgans_gen_type = GenerativeType.ddim
    conf.beta_scheduler = 'linear'
    conf.data_name = 'eeg_encoder'  # Dataset name for encoder
    conf.diffusion_type = 'beatgans'
    conf.eval_ema_every_samples = 200_000
    conf.eval_every_samples = 200_000
    conf.fp16 = True
    conf.lr = 1e-4
    conf.model_name = ModelName.beatgans_autoenc  # Maps to ModelType.autoencoder
    conf.net_attn = (16,)
    conf.net_beatgans_attn_head = 1
    conf.net_beatgans_embed_channels = 512
    conf.net_beatgans_resnet_two_cond = True
    conf.net_ch_mult = (1, 2, 4, 8)
    conf.net_ch = 64
    conf.net_enc_channel_mult = (1, 2, 4, 8, 8)
    conf.net_enc_pool = 'adaptivenonzero'
    conf.sample_size = 32
    conf.T_eval = 20
    conf.T = 1000
    conf.train_mode = TrainMode.diffusion  # Set based on training objective
    conf.make_model_conf()
    conf.name = 'ffhq128_autoenc_130M'
    conf.scale_up_gpus(num_gpus=4)
    return conf

def ffhq128_autoenc_latent():
    """
    Configuration for FFHQ dataset with image size 128 and Latent DPM training.
    """
    conf = TrainConfig()
    conf.batch_size = 32
    conf.beatgans_gen_type = GenerativeType.ddim
    conf.beta_scheduler = 'linear'
    conf.data_name = 'ffhqlmdb256'  # Dataset name for latent diffusion
    conf.diffusion_type = 'beatgans'
    conf.eval_ema_every_samples = 200_000
    conf.eval_every_samples = 200_000
    conf.fp16 = True
    conf.lr = 1e-4
    conf.model_name = ModelName.beatgans_ddpm  # Maps to ModelType.ddpm
    conf.net_attn = (16,)
    conf.net_beatgans_attn_head = 1
    conf.net_beatgans_embed_channels = 512
    conf.net_ch_mult = (1, 1, 2, 3, 4)
    conf.net_ch = 128
    conf.net_enc_channel_mult = (1, 1, 2, 3, 4, 4)
    conf.net_enc_pool = 'adaptivenonzero'
    conf.sample_size = 32
    conf.T_eval = 20
    conf.T = 1000
    conf.train_mode = TrainMode.latent_diffusion  # Set to latent diffusion mode
    conf.scale_up_gpus(num_gpus=1)
    conf.make_model_conf()
    conf.name = 'ffhq128_autoenc_latent'
    return conf


def pretrain_ffhq256_autoenc():
    """
    Pretraining configuration for FFHQ dataset with image size 256 and Autoencoder training.
    """
    conf = ffhq256_autoenc()
    conf.pretrain = PretrainConfig(
        name='90M',
        path=f'checkpoints/{conf.name}/last.ckpt',
    )
    conf.latent_infer_path = f'checkpoints/{conf.name}/latent.pkl'
    return conf


def pretrain_horse128():
    """
    Pretraining configuration for Horse dataset with image size 128 and Autoencoder training.
    """
    conf = horse128_autoenc()
    conf.pretrain = PretrainConfig(
        name='82M',
        path=f'checkpoints/{conf.name}/last.ckpt',
    )
    conf.latent_infer_path = f'checkpoints/{conf.name}/latent.pkl'
    return conf


def pretrain_bedroom128():
    """
    Pretraining configuration for Bedroom dataset with image size 128 and Autoencoder training.
    """
    conf = bedroom128_autoenc()
    conf.pretrain = PretrainConfig(
        name='120M',
        path=f'checkpoints/{conf.name}/last.ckpt',
    )
    conf.latent_infer_path = f'checkpoints/{conf.name}/latent.pkl'
    return conf
