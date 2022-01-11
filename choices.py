from enum import Enum
from torch import nn


class TrainMode(Enum):
    # manipulate mode = training the classifier
    manipulate = 'manipulate'
    # the classifier on the image domain
    manipulate_img = 'manipulateimg'
    # noisy images
    manipulate_imgt = 'manipulateimgt'
    # default trainin mode!
    diffusion = 'diffusion'
    # interpolate the latents while sampling the target images
    diffusion_interpolate = 'interp'
    # using the determistic selection
    diffusion_interpolate_deterministic = 'interpdeterm'
    # weighting less for the samples in the middle
    diffusion_interpolate_deterministic_weight = 'interpdetermw'
    # pow2 weight
    diffusion_interpolate_deterministic_weight_pow2 = 'interpdetermwpow2'
    # only interpolate between closest pairs
    diffusion_interpolate_closest = 'interclose'
    diffusion_interpolate_closest_deterministic = 'interclosedeterm'
    # sampling any image in the batch based on the distance
    diffusion_interpolate_all_img = 'interpallimg'
    autoenc = 'autoenc'
    # also do the diffusion on the latent while training the UNET model
    double_diffusion = 'doublediffusion'
    # default latent training mode!
    # fitting the a DDPM to a given latent
    latent_diffusion = 'latentdiffusion'
    # latent diffusion on 2d latent
    latent_2d_diffusion = 'latent2ddiffusion'
    # latent diffusion is used to condition the Unet, while the condition is the predicted x0 (of the latent)
    # this hopes to address the overelying problem of Unet on the conditioning latent
    parallel_latent_diffusion_pred = 'latentdiffpred'
    # same as the above but with double t sampling
    parallel_latent_diffusion_pred_tt = 'latentdiffpredtt'
    # latent diffusion is used to condition the Unet, the condition is noisy
    parallel_latent_diffusion_noisy = 'latentdiffnoise'
    latent_mmd = 'latent_mmd'
    # a small network predicts the latent and used for the condition
    # but also matched to the encoder for the guidance
    generative_latent = 'genlatent'

    def is_manipulate(self):
        return self in [
            TrainMode.manipulate,
            TrainMode.manipulate_img,
            TrainMode.manipulate_imgt,
        ]

    def is_manipluate_img(self):
        return self in [
            TrainMode.manipulate_img,
            TrainMode.manipulate_imgt,
        ]

    def is_diffusion(self):
        return self in [
            TrainMode.diffusion,
            TrainMode.double_diffusion,
            TrainMode.latent_diffusion,
            TrainMode.diffusion_interpolate,
            TrainMode.diffusion_interpolate_deterministic,
            TrainMode.diffusion_interpolate_deterministic_weight,
            TrainMode.diffusion_interpolate_deterministic_weight_pow2,
            TrainMode.diffusion_interpolate_closest,
            TrainMode.diffusion_interpolate_closest_deterministic,
            TrainMode.diffusion_interpolate_all_img,
        ]

    def is_interpolate(self):
        return self in [
            TrainMode.diffusion_interpolate,
            TrainMode.diffusion_interpolate_deterministic,
            TrainMode.diffusion_interpolate_deterministic_weight,
            TrainMode.diffusion_interpolate_deterministic_weight_pow2,
            TrainMode.diffusion_interpolate_closest,
            TrainMode.diffusion_interpolate_closest_deterministic,
            TrainMode.diffusion_interpolate_all_img,
        ]

    def is_autoenc(self):
        # the network possibly does autoencoding
        return self in [
            TrainMode.diffusion,
            TrainMode.diffusion_interpolate,
            TrainMode.diffusion_interpolate_deterministic,
            TrainMode.diffusion_interpolate_deterministic_weight,
            TrainMode.diffusion_interpolate_deterministic_weight_pow2,
            TrainMode.diffusion_interpolate_closest,
            TrainMode.diffusion_interpolate_closest_deterministic,
            TrainMode.diffusion_interpolate_all_img,
            TrainMode.autoenc,
            TrainMode.generative_latent,
            TrainMode.parallel_latent_diffusion_noisy,
            TrainMode.parallel_latent_diffusion_pred,
            TrainMode.parallel_latent_diffusion_pred_tt,
        ]

    def is_latent_diffusion(self):
        return self in [
            TrainMode.double_diffusion, TrainMode.latent_diffusion,
            TrainMode.latent_2d_diffusion
        ]

    def use_latent_net(self):
        return self.is_latent_diffusion() or self.is_parallel_latent_diffusion(
        )

    def is_parallel_latent_diffusion(self):
        # that is the latent is modeled alongside the unet
        # the unet may be conditioned by the half-baked latent
        return self in [
            TrainMode.parallel_latent_diffusion_pred,
            TrainMode.parallel_latent_diffusion_pred_tt,
            TrainMode.parallel_latent_diffusion_noisy,
        ]

    def require_dataset_infer(self):
        """
        whether training in this mode requires the latent variables to be available?
        """
        # this will precalculate all the latents before hand
        # and the dataset will be all the predicted latents
        return self in [
            TrainMode.latent_diffusion,
            TrainMode.manipulate,
        ]


class ManipulateMode(Enum):
    """
    how to train the classifier to manipulate
    """
    # train on whole celeba attr dataset
    celeba_all = 'all'
    celebahq_all = 'celebahq_all'
    # train on a few show subset
    celeba_fewshot = 'fewshot'
    celeba_fewshot_allneg = 'fewshotallneg'
    # celeba with D2C's crop
    d2c_fewshot = 'd2cfewshot'
    d2c_fewshot_allneg = 'd2cfewshotallneg'
    celebahq_fewshot = 'celebahq_fewshot'
    relighting = 'light'

    def is_celeba_attr(self):
        return self in [
            ManipulateMode.celeba_all,
            ManipulateMode.celeba_fewshot,
            ManipulateMode.celeba_fewshot_allneg,
            ManipulateMode.d2c_fewshot,
            ManipulateMode.d2c_fewshot_allneg,
            ManipulateMode.celebahq_all,
            ManipulateMode.celebahq_fewshot,
        ]

    def is_single_class(self):
        return self in [
            ManipulateMode.celeba_fewshot,
            ManipulateMode.celeba_fewshot_allneg,
            ManipulateMode.d2c_fewshot,
            ManipulateMode.d2c_fewshot_allneg,
            ManipulateMode.celebahq_fewshot,
        ]

    def is_fewshot(self):
        return self in [
            ManipulateMode.celeba_fewshot,
            ManipulateMode.celeba_fewshot_allneg,
            ManipulateMode.d2c_fewshot,
            ManipulateMode.d2c_fewshot_allneg,
            ManipulateMode.celebahq_fewshot,
        ]

    def is_fewshot_allneg(self):
        return self in [
            ManipulateMode.celeba_fewshot_allneg,
            ManipulateMode.d2c_fewshot_allneg,
        ]


class ModelType(Enum):
    """
    Kinds of the backbone models
    """

    # unconditional ddpm
    ddpm = 'ddpm'
    # autoencoding ddpm cannot do unconditional generation
    autoencoder = 'autoencoder'
    # may use CLIP as an encoder
    external_encoder = 'extencoder'
    # unconditional ddpm with vae's encoder
    vaeddpm = 'vaeddpm'
    # with mmd latent loss function
    mmdddpm = 'mmdddpm'
    genlatent = 'genlatent'
    # encoder only
    encoder = 'encoder'

    def has_autoenc(self):
        return self in [
            ModelType.autoencoder,
            ModelType.external_encoder,
            ModelType.vaeddpm,
            ModelType.mmdddpm,
            ModelType.genlatent,
        ]

    def can_sample(self):
        return self in [
            ModelType.ddpm, ModelType.vaeddpm, ModelType.mmdddpm,
            ModelType.genlatent
        ]

    def has_noise_to_cond(self):
        return self in [ModelType.mmdddpm, ModelType.genlatent]


class ChamferType(Enum):
    chamfer = 'chamfer'
    stochastic = 'stochastic'


class ModelName(Enum):
    """
    List of all supported model classes
    """

    default_ddpm = 'default_ddpm'
    default_autoenc = 'default_autoenc'
    default_vaeddpm = 'default_vaeddpm'
    beatgans_ddpm = 'beatgans_ddpm'
    beatgans_autoenc = 'beatgans_autoenc'
    beatgans_mmddpm = 'beatgans_mmddpm'
    beatgans_vaeddpm = 'beatgans_vaeddpm'
    beatgans_gen_latent = 'beatgans_genlatent'
    beatgans_encoder = 'beatgans_encoder'


class EncoderName(Enum):
    """
    List of all encoders for ddpm models
    """

    v1 = 'v1'
    v2 = 'v2'


class ModelMeanType(Enum):
    """
    Which type of output the model predicts.
    """

    prev_x = 'x_prev'  # the model predicts x_{t-1}
    start_x = 'x_start'  # the model predicts x_0
    eps = 'eps'  # the model predicts epsilon
    scaled_start_x = 'scaledxstart'  # the model predicts sqrt(alphacum) x_0


class ModelVarType(Enum):
    """
    What is used as the model's output variance.

    The LEARNED_RANGE option has been added to allow the model to predict
    values between FIXED_SMALL and FIXED_LARGE, making its job easier.
    """

    # learned directly
    learned = 'learned'
    # posterior beta_t
    fixed_small = 'fixed_small'
    # beta_t
    fixed_large = 'fixed_large'
    # predict values between FIXED_SMALL and FIXED_LARGE, making its job easier
    learned_range = 'learned_range'


class LossType(Enum):
    mse = 'mse'  # use raw MSE loss (and KL when learning variances)
    l1 = 'l1'
    # mse weighted by the variance, somewhat like in kl
    mse_var_weighted = 'mse_weighted'
    mse_rescaled = 'mse_rescaled'  # use raw MSE loss (with RESCALED_KL when learning variances)
    kl = 'kl'  # use the variational lower-bound
    kl_rescaled = 'kl_rescaled'  # like KL, but rescale to estimate the full VLB

    def is_vb(self):
        return self == LossType.kl or self == LossType.kl_rescaled


class MSEWeightType(Enum):
    # use the ddpm's default variance (either analytical or learned)
    var = 'var'
    # optimal variance by deriving the min kl per image (based on mse of epsilon)
    # = small sigma + mse
    var_min_kl_img = 'varoptimg'
    # optimal variance regradless of the posterior sigmas
    # = mse only
    var_min_kl_mse_img = 'varoptmseimg'
    # same as the above but is based on mse of mu of xprev
    var_min_kl_xprev_img = 'varoptxprevimg'


class XStartWeightType(Enum):
    # weights for the mse of the xstart
    # unweighted x start
    uniform = 'uniform'
    # reciprocal 1 - alpha_bar
    reciprocal_alphabar = 'recipalpha'
    # same as the above but not exceeding mse = 1
    reciprocal_alphabar_safe = 'recipalphasafe'
    # turning x0 into eps as use the mse(eps)
    eps = 'eps'
    # the same as above but not turning into eps
    eps2 = 'eps2'
    # same as the above but not exceeding mse = 1
    eps2_safe = 'eps2safe'
    eps_huber = 'epshuber'
    unit_mse_x0 = 'unitmsex0'
    unit_mse_eps = 'unitmseeps'


class GenerativeType(Enum):
    """
    How's a sample generated
    """

    ddpm = 'ddpm'
    ddim = 'ddim'


class OptimizerType(Enum):
    adam = 'adam'
    adamw = 'adamw'


class ConditionType(Enum):
    no = 'no'
    add = 'add'
    scale_shift_norm = 'normmod'
    scale_shift_hybrid = 'normhybrid'
    # this option only available on MLPs
    norm_scale_shift = 'normmodafter'


class Activation(Enum):
    none = 'none'
    relu = 'relu'
    lrelu = 'lrelu'
    silu = 'silu'
    tanh = 'tanh'

    def get_act(self):
        if self == Activation.none:
            return nn.Identity()
        elif self == Activation.relu:
            return nn.ReLU()
        elif self == Activation.lrelu:
            return nn.LeakyReLU(negative_slope=0.2)
        elif self == Activation.silu:
            return nn.SiLU()
        elif self == Activation.tanh:
            return nn.Tanh()
        else:
            raise NotImplementedError()


class CondAt(Enum):
    """
    which layers recieve the encoder's conditions
    """
    all = 'all'
    enc = 'enc'
    mid_dec = 'middec'
    dec = 'dec'


class ManipulateLossType(Enum):
    bce = 'bce'
    mse = 'mse'