
import copy
import json
import os
import re
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, TensorDataset
from torchvision.utils import make_grid, save_image
from contextlib import nullcontext

from config import *
from dataset import ConditionalEEGImageDataset  # Updated dataset class
from dist_utils import get_world_size, get_rank
from metrics import evaluate_fid, evaluate_lpips
from renderer import render_condition, render_uncondition

# Configure logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class LitModel(pl.LightningModule):
    def __init__(self, conf: TrainConfig):
        super().__init__()
        assert conf.train_mode != TrainMode.manipulate, "Manipulate mode is not supported."

        if conf.seed is not None:
            pl.seed_everything(conf.seed)

        self.save_hyperparameters(conf.as_dict_jsonable())

        self.conf = conf

        # Initialize the main model and EMA model
        self.model = conf.make_model_conf().make_model()
        self.ema_model = copy.deepcopy(self.model)
        self.ema_model.requires_grad_(False)
        self.ema_model.eval()

        # Calculate and log model size
        model_size = sum(param.numel() for param in self.model.parameters())
        logging.info(f'Model parameters: {model_size / 1e6:.2f} M')

        # Initialize samplers
        self.sampler = conf.make_diffusion_conf().make_sampler()
        self.eval_sampler = conf.make_eval_diffusion_conf().make_sampler()

        if conf.train_mode.use_latent_net():
            self.latent_sampler = conf.make_latent_diffusion_conf().make_sampler()
            self.eval_latent_sampler = conf.make_latent_eval_diffusion_conf().make_sampler()
        else:
            self.latent_sampler = None
            self.eval_latent_sampler = None

        # Register a buffer for consistent sampling
        self.register_buffer('x_T', torch.randn(conf.sample_size, 3, conf.img_size, conf.img_size))

        # Load pre-trained weights if specified
        if conf.pretrain is not None:
            logging.info(f'Loading pre-trained model from {conf.pretrain.name}')
            state = torch.load(conf.pretrain.path, map_location='cpu')
            logging.info(f'Checkpoint loaded at step {state["global_step"]}')
            self.load_state_dict(state['state_dict'], strict=False)

        # Load latent statistics if specified
        if conf.latent_infer_path is not None:
            logging.info('Loading latent statistics...')
            state = torch.load(conf.latent_infer_path, map_location='cpu')
            self.conds = state['conds']
            self.register_buffer('conds_mean', state['conds_mean'][None, :])
            self.register_buffer('conds_std', state['conds_std'][None, :])
        else:
            self.conds_mean = None
            self.conds_std = None

    def normalize(self, cond):
        return (cond - self.conds_mean.to(self.device)) / self.conds_std.to(self.device)

    def denormalize(self, cond):
        return (cond * self.conds_std.to(self.device)) + self.conds_mean.to(self.device)

    def sample(self, N, device, T=None, T_latent=None):
        if T is None:
            sampler = self.eval_sampler
            latent_sampler = self.latent_sampler
        else:
            sampler = self.conf._make_diffusion_conf(T).make_sampler()
            latent_sampler = self.conf._make_latent_diffusion_conf(T_latent).make_sampler()

        noise = torch.randn(N, 3, self.conf.img_size, self.conf.img_size, device=device)
        pred_img = render_uncondition(
            self.conf,
            self.ema_model,
            noise,
            sampler=sampler,
            latent_sampler=latent_sampler,
            conds_mean=self.conds_mean,
            conds_std=self.conds_std,
        )
        pred_img = (pred_img + 1) / 2
        return pred_img

    def render(self, noise, cond=None, T=None):
        if T is None:
            sampler = self.eval_sampler
        else:
            sampler = self.conf._make_diffusion_conf(T).make_sampler()

        if cond is not None:
            pred_img = render_condition(
                self.conf,
                self.ema_model,
                noise,
                sampler=sampler,
                cond=cond
            )
        else:
            pred_img = render_uncondition(
                self.conf,
                self.ema_model,
                noise,
                sampler=sampler,
                latent_sampler=None
            )
        pred_img = (pred_img + 1) / 2
        return pred_img

    def encode(self, x):
        assert self.conf.model_type.has_autoenc(), "Model does not have an autoencoder."
        return self.ema_model.encoder(x)

    def encode_stochastic(self, x, cond, T=None):
        if T is None:
            sampler = self.eval_sampler
        else:
            sampler = self.conf._make_diffusion_conf(T).make_sampler()
        out = sampler.ddim_reverse_sample_loop(
            self.ema_model,
            x,
            model_kwargs={'cond': cond}
        )
        return out['sample']

    def forward(self, noise=None, x_start=None, ema_model: bool = False):
        with torch.cuda.amp.autocast(False):
            model = self.ema_model if ema_model else self.model
            gen = self.eval_sampler.sample(model=model, noise=noise, x_start=x_start)
            return gen

    def setup(self, stage=None) -> None:
        """
        Initialize datasets and ensure each worker has a unique seed.
        """
        if self.conf.seed is not None:
            seed = self.conf.seed * get_world_size() + get_rank()
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            logging.info(f'Local seed: {seed}')

        # Load training and validation datasets based on train mode
        if self.conf.train_mode == TrainMode.encoder:
            self.train_data = self.conf.make_dataset()  # Should point to 'eeg_encoder.lmdb'
            self.val_data = self.conf.make_validation_dataset()  # Optional separate validation
        elif self.conf.train_mode == TrainMode.latent_diffusion:
            self.train_data = self.conf.make_dataset()  # Should point to 'eeg_ddim_ffhq.lmdb'
            self.val_data = self.conf.make_validation_dataset()  # Optional separate validation
        else:
            raise NotImplementedError("Unsupported train mode.")

        logging.info(f'Training data size: {len(self.train_data)}')
        logging.info(f'Validation data size: {len(self.val_data)}')

    def _train_dataloader(self, drop_last=True):
        """
        Create a DataLoader for training.
        """
        conf = self.conf.clone()
        conf.batch_size = self.batch_size
        dataloader = conf.make_loader(
            self.train_data,
            shuffle=True,
            drop_last=drop_last
        )
        return dataloader

    def train_dataloader(self):
        """
        Return the appropriate DataLoader based on the training mode.
        """
        logging.info('Initializing training dataloader...')
        if self.conf.train_mode == TrainMode.encoder:
            # Encoder training: only EEG data
            conf = self.conf.clone()
            conf.batch_size = self.batch_size
            data = TensorDataset(self.conds)
            return conf.make_loader(data, shuffle=True)
        elif self.conf.train_mode == TrainMode.latent_diffusion:
            # Conditional DDIM training: EEG and Image data
            return self._train_dataloader()
        else:
            raise NotImplementedError("Unsupported train mode.")

    @property
    def batch_size(self):
        """
        Calculate local batch size based on world size.
        """
        ws = get_world_size()
        assert self.conf.batch_size % ws == 0, "Batch size must be divisible by world size."
        return self.conf.batch_size // ws

    @property
    def num_samples(self):
        """
        Calculate the total number of samples processed.
        """
        return self.global_step * self.conf.batch_size_effective

    def is_last_accum(self, batch_idx):
        """
        Check if it's the last gradient accumulation step.
        """
        return (batch_idx + 1) % self.conf.accum_batches == 0

    def infer_whole_dataset(self, with_render=False, T_render=None, render_save_path=None):
        """
        Encode the entire dataset to obtain latent representations.

        Args:
            with_render (bool): Whether to render images.
            T_render (int): Diffusion steps for rendering.
            render_save_path (str): Path to save rendered images.
        """
        data = self.conf.make_dataset()
        data.transform = make_transform(self.conf.img_size, flip_prob=0)

        loader = self.conf.make_loader(
            data,
            shuffle=False,
            drop_last=False,
            batch_size=self.conf.batch_size_eval,
            parallel=True,
        )
        model = self.ema_model
        model.eval()
        conds = []

        if with_render:
            sampler = self.conf._make_diffusion_conf(T=T_render or self.conf.T_eval).make_sampler()
            if get_rank() == 0:
                writer = LMDBImageWriter(render_save_path, format='webp', quality=100)
            else:
                writer = nullcontext()
        else:
            writer = nullcontext()

        with writer:
            for batch in tqdm(loader, total=len(loader), desc='Infer'):
                with torch.no_grad():
                    imgs = batch['img'].to(self.device)
                    cond = model.encoder(imgs)

                    # Optional normalization
                    if self.conf.latent_znormalize:
                        cond = self.normalize(cond)

                    # Gather all ranks
                    idx = batch['index']
                    idx = self.all_gather(idx)
                    if idx.dim() == 2:
                        idx = idx.flatten(0, 1)
                    argsort = idx.argsort()

                    if with_render:
                        noise = torch.randn(len(cond), 3, self.conf.img_size, self.conf.img_size, device=self.device)
                        render = sampler.sample(
                            model=model,
                            noise=noise,
                            cond=cond
                        )
                        render = (render + 1) / 2
                        render = self.all_gather(render)
                        if render.dim() == 5:
                            render = render.flatten(0, 1)

                        if get_rank() == 0:
                            writer.put_images(render[argsort])

                    cond = self.all_gather(cond)
                    if cond.dim() == 3:
                        cond = cond.flatten(0, 1)

                    conds.append(cond[argsort].cpu())

        model.train()
        conds = torch.cat(conds).float()
        return conds

    def training_step(self, batch, batch_idx):
        """
        Define the training logic for both encoder and conditional DDIM.
        """
        with torch.cuda.amp.autocast(False):
            if self.conf.train_mode == TrainMode.encoder:
                # Training Semantic Encoder
                eeg = batch[0].to(self.device)  # Assuming batch is TensorDataset of EEG
                encoded = self.model.encoder(eeg)
                # If there's a decoder, compute reconstruction loss
                if hasattr(self.model, 'decoder') and self.model.decoder is not None:
                    reconstructed = self.model.decoder(encoded)
                    loss = nn.MSELoss()(reconstructed, eeg)
                else:
                    # Define a suitable loss if no decoder exists
                    loss = torch.mean(encoded)  # Example placeholder
                self.log('encoder_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

            elif self.conf.train_mode == TrainMode.latent_diffusion:
                # Training Conditional DDIM
                eeg, img = batch  # Assuming batch is (EEG, Image)
                eeg = eeg.to(self.device)
                img = img.to(self.device)

                # Encode EEG to get z_sem
                z_sem = self.model.encoder(eeg)

                # Normalize if required
                if self.conf.latent_znormalize:
                    z_sem = self.normalize(z_sem)

                # Sample diffusion timesteps
                t, weight = self.T_sampler.sample(len(z_sem), z_sem.device)

                # Compute diffusion loss conditioned on z_sem
                losses = self.sampler.training_losses(
                    model=self.model,
                    x_start=img,
                    t=t,
                    cond=z_sem
                )

                loss = losses['loss'].mean()
                self.log('ddim_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

            else:
                raise NotImplementedError("Unsupported training mode.")

        return {'loss': loss}

    def on_train_batch_end(self, outputs, batch, batch_idx: int, dataloader_idx: int) -> None:
        """
        Handle EMA updates and logging after each training batch.
        """
        if self.is_last_accum(batch_idx):
            # Update EMA
            if self.conf.train_mode == TrainMode.latent_diffusion:
                ema(self.model.latent_net, self.ema_model.latent_net, self.conf.ema_decay)
            else:
                ema(self.model, self.ema_model, self.conf.ema_decay)

            # Log samples and evaluate scores
            if self.conf.train_mode == TrainMode.encoder:
                self.log_sample(x_start=None)  # Encoder does not require x_start
            elif self.conf.train_mode == TrainMode.latent_diffusion:
                self.log_sample(x_start=None)  # Conditional DDIM will handle conditioning
            self.evaluate_scores()

    def on_before_optimizer_step(self, optimizer: Optimizer, optimizer_idx: int) -> None:
        """
        Apply gradient clipping before the optimizer step.
        """
        if self.conf.grad_clip > 0:
            params = [p for group in optimizer.param_groups for p in group['params']]
            torch.nn.utils.clip_grad_norm_(params, max_norm=self.conf.grad_clip)

    def log_sample(self, x_start):
        """
        Log generated samples to TensorBoard.
        """
        def do(model, postfix, use_xstart, save_real=False, no_latent_diff=False, interpolate=False):
            model.eval()
            with torch.no_grad():
                all_x_T = self.split_tensor(self.x_T)
                batch_size = min(len(all_x_T), self.conf.batch_size_eval)
                loader = DataLoader(all_x_T, batch_size=batch_size)

                Gen = []
                for x_T in loader:
                    if use_xstart:
                        _xstart = x_start[:len(x_T)]
                    else:
                        _xstart = None

                    if self.conf.train_mode.is_latent_diffusion() and not use_xstart:
                        # Conditional DDIM sampling
                        gen = render_uncondition(
                            conf=self.conf,
                            model=model,
                            x_T=x_T.to(self.device),
                            sampler=self.eval_sampler,
                            latent_sampler=self.eval_latent_sampler,
                            conds_mean=self.conds_mean,
                            conds_std=self.conds_std,
                        )
                    else:
                        # Unconditional or autoencoding sampling
                        if not use_xstart and self.conf.model_type.has_noise_to_cond():
                            cond = torch.randn(len(x_T), self.conf.style_ch, device=self.device)
                            cond = model.noise_to_cond(cond)
                        else:
                            if interpolate:
                                with torch.cuda.amp.autocast(self.conf.fp16):
                                    cond = model.encoder(_xstart)
                                    i = torch.randperm(len(cond))
                                    cond = (cond + cond[i]) / 2
                            else:
                                cond = None
                        gen = self.eval_sampler.sample(
                            model=model,
                            noise=x_T.to(self.device),
                            cond=cond,
                            x_start=_xstart
                        )
                    Gen.append(gen)

                gen = torch.cat(Gen)
                gen = self.all_gather(gen)
                if gen.dim() == 5:
                    gen = gen.flatten(0, 1)

                if save_real and use_xstart:
                    real = self.all_gather(_xstart)
                    if real.dim() == 5:
                        real = real.flatten(0, 1)
                    if get_rank() == 0:
                        grid_real = (make_grid(real) + 1) / 2
                        self.logger.experiment.add_image(f'sample{postfix}/real', grid_real, self.num_samples)

                if get_rank() == 0:
                    grid = (make_grid(gen) + 1) / 2
                    sample_dir = os.path.join(self.conf.logdir, f'sample{postfix}')
                    os.makedirs(sample_dir, exist_ok=True)
                    path = os.path.join(sample_dir, f'{self.num_samples}.png')
                    save_image(grid, path)
                    self.logger.experiment.add_image(f'sample{postfix}', grid, self.num_samples)
            model.train()

        if self.conf.sample_every_samples > 0 and is_time(
                self.num_samples, self.conf.sample_every_samples, self.conf.batch_size_effective):

            if self.conf.train_mode == TrainMode.encoder:
                do(self.model, '', use_xstart=False)
                do(self.ema_model, '_ema', use_xstart=False)
            elif self.conf.train_mode == TrainMode.latent_diffusion:
                do(self.model, '', use_xstart=False)
                do(self.ema_model, '_ema', use_xstart=False)
                # Conditional DDIM might include additional sampling logic
            else:
                do(self.model, '', use_xstart=True, save_real=True)
                do(self.ema_model, '_ema', use_xstart=True, save_real=True)

    def evaluate_scores(self):
        """
        Evaluate metrics like FID and LPIPS during training.
        """
        def fid(model, postfix):
            score = evaluate_fid(
                sampler=self.eval_sampler,
                model=model,
                conf=self.conf,
                device=self.device,
                train_data=self.train_data,
                val_data=self.val_data,
                latent_sampler=self.eval_latent_sampler,
                conds_mean=self.conds_mean,
                conds_std=self.conds_std
            )
            if get_rank() == 0:
                self.logger.experiment.add_scalar(f'FID{postfix}', score, self.num_samples)
                os.makedirs(self.conf.logdir, exist_ok=True)
                with open(os.path.join(self.conf.logdir, 'eval.txt'), 'a') as f:
                    metrics = {
                        f'FID{postfix}': score,
                        'num_samples': self.num_samples,
                    }
                    f.write(json.dumps(metrics) + "\n")

        def lpips(model, postfix):
            if self.conf.model_type.has_autoenc() and self.conf.train_mode.is_autoenc():
                # {'lpips', 'ssim', 'mse'}
                score = evaluate_lpips(
                    sampler=self.eval_sampler,
                    model=model,
                    conf=self.conf,
                    device=self.device,
                    val_data=self.val_data,
                    latent_sampler=self.eval_latent_sampler
                )
                if get_rank() == 0:
                    for key, val in score.items():
                        self.logger.experiment.add_scalar(f'{key}{postfix}', val, self.num_samples)

        if self.conf.eval_every_samples > 0 and self.num_samples > 0 and is_time(
                self.num_samples, self.conf.eval_every_samples, self.conf.batch_size_effective):
            logging.info(f'Evaluating FID at step {self.num_samples}')
            lpips(self.model, '')
            fid(self.model, '')

        if self.conf.eval_ema_every_samples > 0 and self.num_samples > 0 and is_time(
                self.num_samples, self.conf.eval_ema_every_samples, self.conf.batch_size_effective):
            logging.info(f'Evaluating FID EMA at step {self.num_samples}')
            fid(self.ema_model, '_ema')
            # LPIPS for EMA is commented out due to speed
            # lpips(self.ema_model, '_ema')

    def configure_optimizers(self):
        """
        Configure optimizers and learning rate schedulers.
        """
        if self.conf.train_mode == TrainMode.encoder:
            # Optimizer for Semantic Encoder
            if self.conf.optimizer == OptimizerType.adam:
                optim = torch.optim.Adam(self.model.encoder.parameters(),
                                         lr=self.conf.lr,
                                         weight_decay=self.conf.weight_decay)
            elif self.conf.optimizer == OptimizerType.adamw:
                optim = torch.optim.AdamW(self.model.encoder.parameters(),
                                          lr=self.conf.lr,
                                          weight_decay=self.conf.weight_decay)
            else:
                raise NotImplementedError("Unsupported optimizer type.")
        elif self.conf.train_mode == TrainMode.latent_diffusion:
            # Optimizer for Conditional DDIM (latent_net)
            if self.conf.optimizer == OptimizerType.adam:
                optim = torch.optim.Adam(self.model.latent_net.parameters(),
                                         lr=self.conf.lr,
                                         weight_decay=self.conf.weight_decay)
            elif self.conf.optimizer == OptimizerType.adamw:
                optim = torch.optim.AdamW(self.model.latent_net.parameters(),
                                          lr=self.conf.lr,
                                          weight_decay=self.conf.weight_decay)
            else:
                raise NotImplementedError("Unsupported optimizer type.")
        else:
            raise NotImplementedError("Unsupported train mode.")

        optimizer_config = {'optimizer': optim}

        if self.conf.warmup > 0:
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optim,
                lr_lambda=WarmupLR(self.conf.warmup)
            )
            optimizer_config['lr_scheduler'] = {
                'scheduler': scheduler,
                'interval': 'step',
            }

        return optimizer_config

    def split_tensor(self, x):
        """
        Extract the tensor corresponding to the current worker in distributed training.

        Args:
            x (torch.Tensor): Input tensor of shape (n, c).

        Returns:
            torch.Tensor: Sub-tensor for the current worker.
        """
        n = len(x)
        rank = get_rank()
        world_size = get_world_size()
        per_rank = n // world_size
        return x[rank * per_rank:(rank + 1) * per_rank]

    def test_step(self, batch, *args, **kwargs):
        """
        Handle evaluation tasks based on `conf.eval_programs`.
        """
        # Ensure each worker has a unique seed
        self.setup()

        logging.info(f'Global step: {self.global_step}')

        if 'infer' in self.conf.eval_programs:
            logging.info('Performing inference...')
            conds = self.infer_whole_dataset().float()
            save_path = f'checkpoints/{self.conf.name}/latent.pkl'

            if get_rank() == 0:
                conds_mean = conds.mean(dim=0)
                conds_std = conds.std(dim=0)
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save({
                    'conds': conds,
                    'conds_mean': conds_mean,
                    'conds_std': conds_std,
                }, save_path)

        for each in self.conf.eval_programs:
            if each.startswith('infer+render'):
                m = re.match(r'infer\+render([0-9]+)', each)
                if m is not None:
                    T = int(m[1])
                    logging.info(f'Performing infer + render with T={T}...')
                    conds = self.infer_whole_dataset(
                        with_render=True,
                        T_render=T,
                        render_save_path=f'latent_infer_render{T}/{self.conf.name}.lmdb',
                    )
                    save_path = f'latent_infer_render{T}/{self.conf.name}.pkl'
                    conds_mean = conds.mean(dim=0)
                    conds_std = conds.std(dim=0)
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    torch.save({
                        'conds': conds,
                        'conds_mean': conds_mean,
                        'conds_std': conds_std,
                    }, save_path)

        for each in self.conf.eval_programs:
            if each.startswith('fid'):
                m = re.match(r'fid\(([0-9]+),([0-9]+)\)', each)
                clip_latent_noise = False
                if m is not None:
                    T = int(m[1])
                    T_latent = int(m[2])
                    logging.info(f'Evaluating FID with T={T}, latent T={T_latent}...')
                else:
                    m = re.match(r'fidclip\(([0-9]+),([0-9]+)\)', each)
                    if m is not None:
                        T = int(m[1])
                        T_latent = int(m[2])
                        clip_latent_noise = True
                        logging.info(f'Evaluating FID (clip latent noise) with T={T}, latent T={T_latent}...')
                    else:
                        _, T = each.split('fid')
                        T = int(T)
                        T_latent = None
                        logging.info(f'Evaluating FID with T={T}...')

                self.train_dataloader()
                sampler = self.conf._make_diffusion_conf(T=T).make_sampler()
                latent_sampler = self.conf._make_latent_diffusion_conf(T_latent).make_sampler() if T_latent else None

                conf = self.conf.clone()
                conf.eval_num_images = 50_000
                score = evaluate_fid(
                    sampler=sampler,
                    model=self.ema_model,
                    conf=conf,
                    device=self.device,
                    train_data=self.train_data,
                    val_data=self.val_data,
                    latent_sampler=latent_sampler,
                    conds_mean=self.conds_mean,
                    conds_std=self.conds_std,
                    remove_cache=False,
                    clip_latent_noise=clip_latent_noise,
                )
                if T_latent is None:
                    self.log(f'fid_ema_T{T}', score)
                else:
                    name = 'fid_clip' if clip_latent_noise else 'fid'
                    name += f'_ema_T{T}_Tlatent{T_latent}'
                    self.log(name, score)

            elif each.startswith('recon'):
                _, T = each.split('recon')
                T = int(T)
                logging.info(f'Evaluating reconstruction with T={T}...')
                sampler = self.conf._make_diffusion_conf(T=T).make_sampler()

                conf = self.conf.clone()
                conf.eval_num_images = len(self.val_data)
                score = evaluate_lpips(
                    sampler=sampler,
                    model=self.ema_model,
                    conf=conf,
                    device=self.device,
                    val_data=self.val_data,
                    latent_sampler=None
                )
                for k, v in score.items():
                    self.log(f'{k}_ema_T{T}', v)

            elif each.startswith('inv'):
                _, T = each.split('inv')
                T = int(T)
                logging.info(f'Evaluating reconstruction with noise inversion T={T}...')
                sampler = self.conf._make_diffusion_conf(T=T).make_sampler()

                conf = self.conf.clone()
                conf.eval_num_images = len(self.val_data)
                score = evaluate_lpips(
                    sampler=sampler,
                    model=self.ema_model,
                    conf=conf,
                    device=self.device,
                    val_data=self.val_data,
                    latent_sampler=None,
                    use_inverted_noise=True
                )
                for k, v in score.items():
                    self.log(f'{k}_inv_ema_T{T}', v)

    def ema(source, target, decay):
        """
        Update the target model's parameters as an exponential moving average of the source model's parameters.
        """
        source_dict = source.state_dict()
        target_dict = target.state_dict()
        for key in source_dict.keys():
            target_dict[key].data.copy_(target_dict[key].data * decay + source_dict[key].data * (1 - decay))

    class WarmupLR:
        """
        Learning rate scheduler with warmup.
        """
        def __init__(self, warmup_steps):
            self.warmup_steps = warmup_steps

        def __call__(self, step):
            return min(step, self.warmup_steps) / self.warmup_steps

    def is_time(num_samples, every, step_size):
        """
        Check if it's time to perform an action based on sampling steps.
        """
        closest = (num_samples // every) * every
        return num_samples - closest < step_size

    def train(conf: TrainConfig, gpus, nodes=1, mode: str = 'train'):
        """
        Main training function to initiate PyTorch Lightning's Trainer.
        """
        logging.info(f'Configuration: {conf.name}')

        model = LitModel(conf)

        os.makedirs(conf.logdir, exist_ok=True)
        checkpoint = ModelCheckpoint(
            dirpath=conf.logdir,
            save_last=True,
            save_top_k=1,
            every_n_train_steps=conf.save_every_samples // conf.batch_size_effective
        )
        checkpoint_path = os.path.join(conf.logdir, 'last.ckpt')
        logging.info(f'Checkpoint path: {checkpoint_path}')

        if os.path.exists(checkpoint_path):
            resume = checkpoint_path
            logging.info('Resuming from checkpoint...')
        else:
            resume = conf.continue_from.path if conf.continue_from is not None else None
            if resume:
                logging.info(f'Resuming from {resume}')
            else:
                logging.info('Starting training from scratch.')

        tb_logger = pl_loggers.TensorBoardLogger(
            save_dir=conf.logdir,
            name=None,
            version=''
        )

        plugins = []
        if len(gpus) > 1 or nodes > 1:
            accelerator = 'ddp'
            from pytorch_lightning.plugins import DDPPlugin
            plugins.append(DDPPlugin(find_unused_parameters=False))
        else:
            accelerator = None

        trainer = pl.Trainer(
            max_steps=conf.total_samples // conf.batch_size_effective,
            resume_from_checkpoint=resume,
            gpus=gpus,
            num_nodes=nodes,
            accelerator=accelerator,
            precision=16 if conf.fp16 else 32,
            callbacks=[
                checkpoint,
                LearningRateMonitor(logging_interval='step'),
            ],
            replace_sampler_ddp=True,
            logger=tb_logger,
            accumulate_grad_batches=conf.accum_batches,
            plugins=plugins,
        )

        if mode == 'train':
            trainer.fit(model)
        elif mode == 'eval':
            # Evaluation mode
            dummy = DataLoader(TensorDataset(torch.zeros(conf.batch_size)), batch_size=conf.batch_size)
            eval_path = conf.eval_path or checkpoint_path
            logging.info(f'Loading checkpoint from: {eval_path}')
            state = torch.load(eval_path, map_location='cpu')
            logging.info(f'Checkpoint step: {state["global_step"]}')
            model.load_state_dict(state['state_dict'])

            out = trainer.test(model, dataloaders=dummy)
            out = out[0]
            print(out)

            if get_rank() == 0:
                # Log to TensorBoard
                for k, v in out.items():
                    tb_logger.experiment.add_scalar(k, v, state['global_step'] * conf.batch_size_effective)

                # Save to file
                tgt = f'evals/{conf.name}.txt'
                os.makedirs(os.path.dirname(tgt), exist_ok=True)
                with open(tgt, 'a') as f:
                    f.write(json.dumps(out) + "\n")
        else:
            raise NotImplementedError("Unsupported mode. Choose 'train' or 'eval'.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train or evaluate the Diffusion Autoencoder model.')
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file (YAML/JSON).')
    parser.add_argument('--gpus', type=int, nargs='+', default=[0], help='List of GPU IDs to use.')
    parser.add_argument('--nodes', type=int, default=1, help='Number of nodes for distributed training.')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval'], help='Mode: train or eval.')

    args = parser.parse_args()

    # Load configuration
    conf = TrainConfig.from_yaml(args.config)

    # Start training or evaluation
    train(conf, gpus=args.gpus, nodes=args.nodes, mode=args.mode)
