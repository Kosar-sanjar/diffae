from config import TrainConfig, PretrainConfig
from choices import TrainMode, ModelType, ModelName, GenerativeType
from experiment import train  # Assuming 'train' is defined in experiment.py


def encoder_base():
    """
    Base configuration for Semantic Encoder models.
    """
    conf = TrainConfig()
    conf.batch_size = 32
    conf.lr = 1e-4
    conf.epochs = 100  # Adjust as needed
    conf.train_mode = TrainMode.diffusion  # Set based on your objective
    conf.model_type = ModelType.autoencoder  # Set model type to autoencoder
    conf.model_name = ModelName.beatgans_autoenc
    conf.data_name = 'eeg_encoder'  # EEG-only dataset
    conf.scale_up_gpus(num_gpus=4)  # Example scaling
    # Set other encoder-specific parameters
    # Example:
    conf.net_ch_mult = (1, 2, 4, 8)
    conf.net_ch = 64
    conf.net_enc_channel_mult = (1, 2, 4, 8, 8)
    conf.net_enc_pool = 'adaptivenonzero'
    conf.sample_size = 32
    conf.T_eval = 20
    conf.T = 1000
    conf.make_model_conf()
    conf.name = 'ffhq128_autoenc_130M'
    return conf


def ddpm_base():
    """
    Base configuration for DDPM-based models.
    """
    conf = TrainConfig()
    conf.batch_size = 32
    conf.lr = 1e-4
    conf.epochs = 100  # Adjust as needed
    conf.train_mode = TrainMode.diffusion  # Default diffusion mode
    conf.model_type = ModelType.ddpm  # Set model type to ddpm
    conf.model_name = ModelName.beatgans_ddpm
    conf.data_name = 'ffhqlmdb256'  # Paired EEG-image dataset
    conf.scale_up_gpus(num_gpus=1)  # Example scaling
    # Set other ddpm-specific parameters
    # Example:
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
    conf.make_model_conf()
    conf.name = 'ffhq128_autoenc_latent'
    return conf


def ffhq128_autoenc_130M():
    """
    Configuration for training the Semantic Encoder with FFHQ dataset (130M samples).
    """
    conf = encoder_base()
    conf.total_samples = 130_000_000
    # Add any additional configurations specific to this setup
    return conf


def ffhq128_autoenc_latent():
    """
    Configuration for training the Latent DPM with FFHQ dataset.
    """
    conf = ddpm_base()
    conf.total_samples = 130_000_000
    # Add any additional configurations specific to this setup
    return conf


# Add other configuration functions as needed
