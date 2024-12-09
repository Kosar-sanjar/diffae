from enum import Enum
from torch import nn


class TrainMode(Enum):
    manipulate = 'manipulate'
    diffusion = 'diffusion'
    latent_diffusion = 'latentdiffusion'

    def is_manipulate(self):
        return self in [
            TrainMode.manipulate,
        ]

    def is_diffusion(self):
        return self in [
            TrainMode.diffusion,
            TrainMode.latent_diffusion,
        ]

    def is_autoenc(self):
        return self in [
            TrainMode.diffusion,
        ]

    def is_latent_diffusion(self):
        return self in [
            TrainMode.latent_diffusion,
        ]

    def use_latent_net(self):
        return self.is_latent_diffusion()

    def require_dataset_infer(self):
        return self in [
            TrainMode.latent_diffusion,
            TrainMode.manipulate,
        ]


class ModelType(Enum):
    ddpm = 'ddpm'
    autoencoder = 'autoencoder'

    def has_autoenc(self):
        return self == ModelType.autoencoder

    def can_sample(self):
        return self in [ModelType.ddpm]
