# experiment.py

import copy
import json
import os
import re

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from tqdm import tqdm  # For progress bars
from torch.utils.data import DataLoader, TensorDataset, Subset
from torchvision.utils import make_grid, save_image
from contextlib import nullcontext

from config import TrainConfig, ModelType, TrainMode, Activation, OptimizerType
from dataset import (
    EEGEncoderDataset,
    ConditionalDDIMDataset,
    FFHQlmdb,
    Horse_lmdb,
    Bedroom_lmdb,
    CelebAlmdb
)
from dist_utils import get_world_size, get_rank
from lmdb_writer import LMDBImageWriter
from metrics import evaluate_fid, evaluate_lpips
from renderer import render_condition, render_uncondition
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning import loggers as pl_loggers
from torch import distributed
from multiprocessing import get_context


class LitModel(pl.LightningModule):
    def __init__(self, conf: TrainConfig):
        super().__init__()
        assert conf.train_mode != TrainMode.manipulate, "Manipulate mode not supported yet."
        if conf.seed is not None:
            pl.seed_everything(conf.seed)

        self.save_hyperparameters(conf.as_dict_jsonable())
        self.conf = conf

        # Initialize the main model based on train_mode
        self.model = conf.make_model_conf().make_model()
        self.ema_model = copy.deepcopy(self.model)
        self.ema_model.requires_grad_(False)
        self.ema_model.eval()

        # Calculate and print model size
        model_size = sum(param.numel() for param in self.model.parameters())
        print(f'Model params: {model_size / 1e6:.2f} M')

        # Initialize samplers
        self.sampler = conf.make_diffusion_conf().make_sampler()
        self.eval_sampler = conf.make_eval_diffusion_conf().make_sampler()

        # Shared sampler for both model and latent
        self.T_sampler = conf.make_T_sampler()

        # Initialize latent samplers if using latent diffusion
        if conf.train_mode.use_latent_net():
            self.latent_sampler = conf.make_latent_diffusion_conf().make_sampler()
            self.eval_latent_sampler = conf.make_latent_eval_diffusion_conf().make_sampler()
        else:
            self.latent_sampler = None
            self.eval_latent_sampler = None

        # Buffer for consistent sampling
        self.register_buffer(
            'x_T',
            torch.randn(conf.sample_size, 3, conf.img_size, conf.img_size)
        )

        # Load pre-trained model if specified
        if conf.pretrain is not None:
            print(f'Loading pretrain from {conf.pretrain.name}...')
            state = torch.load(conf.pretrain.path, map_location='cpu')
            print(f'Loaded pretrain step: {state["global_step"]}')
            self.load_state_dict(state['state_dict'], strict=False)

        # Load latent statistics if available
        if conf.latent_infer_path is not None:
            print('Loading latent statistics...')
            state = torch.load(conf.latent_infer_path)
            self.conds = state['conds']
            self.register_buffer('conds_mean', state['conds_mean'][None, :])
            self.register_buffer('conds_std', state['conds_std'][None, :])
        else:
            self.conds_mean = None
            self.conds_std = None

    def normalize(self, cond):
        """Normalize conditions using mean and std."""
        cond = (cond - self.conds_mean.to(self.device)) / self.conds_std.to(self.device)
        return cond

    def denormalize(self, cond):
        """Denormalize conditions using mean and std."""
        cond = (cond * self.conds_std.to(self.device)) + self.conds_mean.to(self.device)
        return cond

    def sample(self, N, device, T=None, T_latent=None):
        """Generate samples."""
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
        pred_img = (pred_img + 1) / 2  # Normalize to [0,1]
        return pred_img

    def render(self, noise, cond=None, T=None):
        """Render images conditioned on latent variables."""
        if T is None:
            sampler = self.eval_sampler
        else:
            sampler = self.conf._make_diffusion_conf(T).make_sampler()

        if cond is not None:
            pred_img = render_condition(self.conf, self.ema_model, noise, sampler=sampler, cond=cond)
        else:
            pred_img = render_uncondition(self.conf, self.ema_model, noise, sampler=sampler, latent_sampler=None)
        pred_img = (pred_img + 1) / 2  # Normalize to [0,1]
        return pred_img

    def encode(self, x):
        """
        Encode input data into latent conditions using the EMA model's encoder.

        Args:
            x (torch.Tensor): Input data tensor.

        Returns:
            torch.Tensor: Latent conditions.
        """
        if self.conf.train_mode == TrainMode.semantic_encoder:
            assert self.conf.model_type.has_autoenc(), "Model type must support autoencoding for Semantic Encoder."
            cond = self.ema_model.encoder.forward(x)
            return cond
        else:
            raise NotImplementedError("Encode method is only implemented for Semantic Encoder mode.")

    def encode_stochastic(self, x, cond, T=None):
        """
        Perform stochastic encoding with diffusion.

        Args:
            x (torch.Tensor): Input data tensor.
            cond (torch.Tensor): Latent conditions.
            T (int, optional): Number of timesteps.

        Returns:
            torch.Tensor: Encoded samples.
        """
        if T is None:
            sampler = self.eval_sampler
        else:
            sampler = self.conf._make_diffusion_conf(T).make_sampler()
        out = sampler.ddim_reverse_sample_loop(self.ema_model, x, model_kwargs={'cond': cond})
        return out['sample']

    def forward(self, noise=None, x_start=None, ema_model: bool = False):
        """
        Forward pass for unconditional generation.

        Args:
            noise (torch.Tensor, optional): Input noise tensor.
            x_start (torch.Tensor, optional): Starting tensor for diffusion.
            ema_model (bool): Whether to use the EMA model.

        Returns:
            torch.Tensor: Generated samples.
        """
        with torch.cuda.amp.autocast(False):
            model = self.ema_model if ema_model else self.model
            gen = self.eval_sampler.sample(model=model, noise=noise, x_start=x_start)
            return gen

    def setup(self, stage=None) -> None:
        """
        Initialize datasets and set seeds.
        """
        # Seed each worker
        if self.conf.seed is not None:
            seed = self.conf.seed * get_world_size() + get_rank()
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            print(f'Local seed: {seed}')

        # Initialize datasets based on training mode
        if self.conf.train_mode == TrainMode.semantic_encoder:
            self.train_data = self.conf.make_semantic_encoder_dataset(split='train')
            self.val_data = self.conf.make_semantic_encoder_dataset(split='val')
        elif self.conf.train_mode == TrainMode.conditional_ddim:
            self.train_data = self.conf.make_conditional_ddim_dataset(split='train')
            self.val_data = self.conf.make_conditional_ddim_dataset(split='val')
        else:
            self.train_data = self.conf.make_dataset(split='train')
            self.val_data = self.conf.make_dataset(split='val')

        print(f'Train data size: {len(self.train_data)}')
        print(f'Validation data size: {len(self.val_data)}')

    def _train_dataloader(self, drop_last=True):
        """
        Create the training DataLoader.
        """
        conf = self.conf.clone()
        conf.batch_size = self.batch_size

        dataloader = conf.make_loader(
            self.train_data,
            shuffle=True,
            drop_last=drop_last,
            parallel=self.conf.parallel
        )
        return dataloader

    def train_dataloader(self):
        """
        Return the appropriate DataLoader based on training mode.
        """
        print('Initializing training DataLoader...')
        if self.conf.train_mode.require_dataset_infer():
            if self.conf.conds is None:
                # Infer conditions (e.g., latent variables)
                self.conds = self.infer_whole_dataset()
                # Calculate mean and std
                self.conds_mean.data = self.conds.float().mean(dim=0, keepdim=True)
                self.conds_std.data = self.conds.float().std(dim=0, keepdim=True)
            print(f'Condition mean: {self.conds_mean.mean()}, std: {self.conds_std.mean()}')

            # Return the dataset with pre-calculated conditions
            conf = self.conf.clone()
            conf.batch_size = self.batch_size
            data = TensorDataset(self.conds)
            return conf.make_loader(data, shuffle=True, parallel=self.conf.parallel)
        else:
            return self._train_dataloader()

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
        Check if it's the last accumulation step.
        """
        return (batch_idx + 1) % self.conf.accum_batches == 0

    def infer_whole_dataset(self, with_render=False, T_render=None, render_save_path=None):
        """
        Infer latent conditions for the entire dataset.

        Args:
            with_render (bool, optional): Whether to render images.
            T_render (int, optional): Number of timesteps for rendering.
            render_save_path (str, optional): Path to save rendered images.

        Returns:
            torch.Tensor: Inferred conditions.
        """
        data = self.conf.make_dataset(split='train')
        if isinstance(data, CelebAlmdb) and data.crop_d2c:
            # Apply d2c crop transformation
            data.transform = make_transform(self.conf.img_size, flip_prob=0, crop_d2c=True)
        else:
            data.transform = make_transform(self.conf.img_size, flip_prob=0)

        loader = self.conf.make_loader(
            data,
            shuffle=False,
            drop_last=False,
            batch_size=self.conf.batch_size_eval,
            parallel=self.conf.parallel,
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
            for batch in tqdm(loader, total=len(loader), desc='Infer Latent Conditions'):
                with torch.no_grad():
                    if self.conf.train_mode == TrainMode.semantic_encoder:
                        imgs = batch['img'].to(self.device)
                        cond = model.encoder(imgs)
                    elif self.conf.train_mode == TrainMode.conditional_ddim:
                        eeg_data, imgs, _, _ = batch
                        imgs = imgs.to(self.device)
                        cond = model.encoder(imgs)
                    else:
                        raise NotImplementedError()

                    # Reordering to match the original dataset
                    if 'index' in batch:
                        idx = batch['index']
                        idx = self.all_gather(idx)
                        if idx.dim() == 2:
                            idx = idx.flatten(0, 1)
                        argsort = idx.argsort()
                    else:
                        argsort = torch.argsort(torch.arange(len(cond)))

                    if with_render:
                        noise = torch.randn(len(cond), 3, self.conf.img_size, self.conf.img_size, device=self.device)
                        render = sampler.sample(model=model, noise=noise, cond=cond)
                        render = (render + 1) / 2
                        render = self.all_gather(render)
                        if render.dim() == 5:
                            render = render.flatten(0, 1)

                        if get_rank() == 0:
                            writer.put_images(render[argsort])

                    # Gather conditions across all processes
                    cond = self.all_gather(cond)
                    if cond.dim() == 3:
                        cond = cond.flatten(0, 1)

                    conds.append(cond[argsort].cpu())

        model.train()
        conds = torch.cat(conds).float()
        return conds

    def training_step(self, batch, batch_idx):
        """
        Define the training step.
        """
        with torch.cuda.amp.autocast(self.conf.fp16):
            # Determine if conditions are pre-inferred
            if self.conf.train_mode.require_dataset_infer():
                cond = batch[0]
                if self.conf.latent_znormalize:
                    cond = self.normalize(cond)
            else:
                if self.conf.train_mode == TrainMode.semantic_encoder:
                    imgs = batch['img']
                    cond = self.encode(imgs.to(self.device))
                elif self.conf.train_mode == TrainMode.conditional_ddim:
                    eeg_data, imgs, _, _ = batch
                    imgs = imgs.to(self.device)
                    cond = self.encode(imgs)
                else:
                    raise NotImplementedError()

            # Compute losses based on train_mode
            if self.conf.train_mode == TrainMode.semantic_encoder:
                # Example: Reconstruction loss for autoencoder
                reconstructed = self.model.decoder(cond)
                loss = torch.nn.functional.mse_loss(reconstructed, imgs.to(self.device))
                losses = {'loss': loss}
            elif self.conf.train_mode == TrainMode.conditional_ddim:
                # Example: Diffusion loss
                t, weight = self.T_sampler.sample(len(cond), cond.device)
                losses = self.sampler.training_losses(model=self.model, x_start=imgs.to(self.device), t=t, cond=cond)
            else:
                raise NotImplementedError()

            # Aggregate losses
            loss = losses['loss'].mean()
            for key in ['loss', 'vae', 'latent', 'mmd', 'chamfer', 'arg_cnt']:
                if key in losses:
                    losses[key] = self.all_gather(losses[key]).mean()

            # Logging
            if get_rank() == 0:
                self.logger.experiment.add_scalar('loss', losses['loss'], self.num_samples)
                for key in ['vae', 'latent', 'mmd', 'chamfer', 'arg_cnt']:
                    if key in losses:
                        self.logger.experiment.add_scalar(f'loss/{key}', losses[key], self.num_samples)

            return {'loss': loss}

    def on_train_batch_end(self, outputs, batch, batch_idx: int, dataloader_idx: int) -> None:
        """
        Actions after each training batch ends.
        """
        if self.is_last_accum(batch_idx):
            # Apply EMA
            if self.conf.train_mode == TrainMode.latent_diffusion:
                ema(self.model.latent_net, self.ema_model.latent_net, self.conf.ema_decay)
            else:
                ema(self.model, self.ema_model, self.conf.ema_decay)

            # Logging samples and evaluating scores
            if self.conf.train_mode.require_dataset_infer():
                imgs = None
            else:
                if self.conf.train_mode == TrainMode.semantic_encoder:
                    imgs = batch['img']
                elif self.conf.train_mode == TrainMode.conditional_ddim:
                    _, imgs, _, _ = batch
                    imgs = imgs
                else:
                    imgs = batch['img']

            self.log_sample(x_start=imgs)
            self.evaluate_scores()

    def on_before_optimizer_step(self, optimizer: torch.optim.Optimizer, optimizer_idx: int) -> None:
        """
        Clip gradients before the optimizer step.
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
                        _xstart = x_start[:len(x_T)].to(self.device)
                    else:
                        _xstart = None

                    if self.conf.train_mode == TrainMode.semantic_encoder and not use_xstart:
                        # Unconditional generation using Semantic Encoder
                        cond = self.encode(_xstart)
                        gen = self.eval_sampler.sample(model=model, noise=x_T.to(self.device), cond=cond)
                    elif self.conf.train_mode == TrainMode.conditional_ddim:
                        if use_xstart:
                            cond = self.encode(_xstart)
                        else:
                            cond = torch.randn(len(x_T), self.conf.embedding_dim, device=self.device)
                        gen = self.eval_sampler.sample(model=model, noise=x_T.to(self.device), cond=cond)
                    else:
                        raise NotImplementedError()

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

        if self.conf.sample_every_samples > 0 and is_time(self.num_samples, self.conf.sample_every_samples, self.conf.batch_size_effective):
            if self.conf.train_mode.require_dataset_infer():
                do(self.model, '', use_xstart=False)
                do(self.ema_model, '_ema', use_xstart=False)
            else:
                if self.conf.train_mode == TrainMode.semantic_encoder:
                    do(self.model, '', use_xstart=True, save_real=True)
                    do(self.ema_model, '_ema', use_xstart=True, save_real=True)
                elif self.conf.train_mode == TrainMode.conditional_ddim:
                    do(self.model, '', use_xstart=True, save_real=True)
                    do(self.ema_model, '_ema', use_xstart=True, save_real=True)
                else:
                    raise NotImplementedError()

    def evaluate_scores(self):
        """
        Evaluate metrics like FID and LPIPS during training.
        """
        def fid(model, postfix):
            score = evaluate_fid(
                self.eval_sampler,
                model,
                self.conf,
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

        def lpips_evaluate(model, postfix):
            if self.conf.model_type.has_autoenc() and self.conf.train_mode == TrainMode.semantic_encoder:
                # Evaluate LPIPS, SSIM, MSE
                score = evaluate_lpips(
                    self.eval_sampler,
                    model,
                    self.conf,
                    device=self.device,
                    val_data=self.val_data,
                    latent_sampler=self.eval_latent_sampler
                )

                if get_rank() == 0:
                    for key, val in score.items():
                        self.logger.experiment.add_scalar(f'{key}{postfix}', val, self.num_samples)

        # Evaluate regular metrics
        if self.conf.eval_every_samples > 0 and self.num_samples > 0 and is_time(self.num_samples, self.conf.eval_every_samples, self.conf.batch_size_effective):
            print(f'Evaluating FID at step {self.num_samples}...')
            lpips_evaluate(self.model, '')
            fid(self.model, '')

        # Evaluate EMA metrics
        if self.conf.eval_ema_every_samples > 0 and self.num_samples > 0 and is_time(self.num_samples, self.conf.eval_ema_every_samples, self.conf.batch_size_effective):
            print(f'Evaluating FID EMA at step {self.num_samples}...')
            fid(self.ema_model, '_ema')
            # Optionally evaluate LPIPS for EMA model
            # lpips_evaluate(self.ema_model, '_ema')  # Uncomment if needed

    def configure_optimizers(self):
        """
        Configure optimizers and learning rate schedulers.
        """
        if self.conf.optimizer == OptimizerType.adam:
            optim = torch.optim.Adam(self.model.parameters(), lr=self.conf.lr, weight_decay=self.conf.weight_decay)
        elif self.conf.optimizer == OptimizerType.adamw:
            optim = torch.optim.AdamW(self.model.parameters(), lr=self.conf.lr, weight_decay=self.conf.weight_decay)
        else:
            raise NotImplementedError(f"Optimizer {self.conf.optimizer} not implemented.")

        if self.conf.warmup > 0:
            sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=WarmupLR(self.conf.warmup))
            return {
                'optimizer': optim,
                'lr_scheduler': {
                    'scheduler': sched,
                    'interval': 'step',
                }
            }
        else:
            return {'optimizer': optim}

    def split_tensor(self, x):
        """
        Split tensor for distributed training.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Split tensor for the current process.
        """
        n = len(x)
        rank = get_rank()
        world_size = get_world_size()
        per_rank = n // world_size
        return x[rank * per_rank:(rank + 1) * per_rank]

    def test_step(self, batch, *args, **kwargs):
        """
        Define the test step for evaluation.
        """
        # Initialize datasets and models
        self.setup()

        print(f'Global step during test: {self.global_step}')

        # Handle different evaluation programs
        for program in self.conf.eval_programs:
            if program == 'infer':
                print('Running infer...')
                conds = self.infer_whole_dataset().float()
                save_path = f'checkpoints/{self.conf.name}/latent.pkl'

                if get_rank() == 0:
                    conds_mean = conds.mean(dim=0)
                    conds_std = conds.std(dim=0)
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    torch.save(
                        {
                            'conds': conds,
                            'conds_mean': conds_mean,
                            'conds_std': conds_std,
                        },
                        save_path
                    )

            elif program.startswith('infer+render'):
                m = re.match(r'infer\+render([0-9]+)', program)
                if m:
                    T = int(m.group(1))
                    print(f'Running infer+render with T={T}...')
                    conds = self.infer_whole_dataset(
                        with_render=True,
                        T_render=T,
                        render_save_path=f'latent_infer_render{T}/{self.conf.name}.lmdb',
                    )
                    save_path = f'latent_infer_render{T}/{self.conf.name}.pkl'
                    conds_mean = conds.mean(dim=0)
                    conds_std = conds.std(dim=0)
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    torch.save(
                        {
                            'conds': conds,
                            'conds_mean': conds_mean,
                            'conds_std': conds_std,
                        },
                        save_path
                    )

            elif program.startswith('fid'):
                m = re.match(r'fid\(([0-9]+),([0-9]+)\)', program)
                clip_latent_noise = False
                if m:
                    T = int(m.group(1))
                    T_latent = int(m.group(2))
                    print(f'Evaluating FID with T={T} and T_latent={T_latent}...')
                else:
                    m = re.match(r'fidclip\(([0-9]+),([0-9]+)\)', program)
                    if m:
                        T = int(m.group(1))
                        T_latent = int(m.group(2))
                        clip_latent_noise = True
                        print(f'Evaluating FID with clip latent noise, T={T}, T_latent={T_latent}...')
                    else:
                        # Assume format 'fidT'
                        _, T = program.split('fid')
                        T = int(T)
                        T_latent = None
                        print(f'Evaluating FID with T={T}...')

                sampler = self.conf._make_diffusion_conf(T=T).make_sampler()
                latent_sampler = self.conf._make_latent_diffusion_conf(T=T_latent).make_sampler() if T_latent else None

                conf = self.conf.clone()
                conf.eval_num_images = 50000  # Adjust as needed

                score = evaluate_fid(
                    sampler,
                    self.ema_model,
                    conf,
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
                    name = 'fid'
                    if clip_latent_noise:
                        name += '_clip'
                    name += f'_ema_T{T}_Tlatent{T_latent}'
                    self.log(name, score)

            elif program.startswith('recon'):
                _, T = program.split('recon')
                T = int(T)
                print(f'Evaluating reconstruction with T={T}...')

                sampler = self.conf._make_diffusion_conf(T=T).make_sampler()

                conf = self.conf.clone()
                conf.eval_num_images = len(self.val_data)

                score = evaluate_lpips(
                    sampler,
                    self.ema_model,
                    conf,
                    device=self.device,
                    val_data=self.val_data,
                    latent_sampler=None
                )

                for k, v in score.items():
                    self.log(f'{k}_ema_T{T}', v)

            elif program.startswith('inv'):
                _, T = program.split('inv')
                T = int(T)
                print(f'Evaluating reconstruction with noise inversion T={T}...')

                sampler = self.conf._make_diffusion_conf(T=T).make_sampler()

                conf = self.conf.clone()
                conf.eval_num_images = len(self.val_data)

                score = evaluate_lpips(
                    sampler,
                    self.ema_model,
                    conf,
                    device=self.device,
                    val_data=self.val_data,
                    latent_sampler=None,
                    use_inverted_noise=True
                )

                for k, v in score.items():
                    self.log(f'{k}_inv_ema_T{T}', v)

    def ema(source, target, decay):
        """
        Update EMA model parameters.
        """
        source_dict = source.state_dict()
        target_dict = target.state_dict()
        for key in source_dict.keys():
            target_dict[key].data.copy_(target_dict[key].data * decay + source_dict[key].data * (1 - decay))

    class WarmupLR:
        def __init__(self, warmup_steps: int) -> None:
            self.warmup_steps = warmup_steps

        def __call__(self, step: int):
            return min(step, self.warmup_steps) / self.warmup_steps

    def is_time(num_samples, every, step_size):
        """
        Check if the current step is time to perform an action based on 'every'.

        Args:
            num_samples (int): Total samples processed.
            every (int): Interval at which to perform the action.
            step_size (int): Number of samples per step.

        Returns:
            bool: Whether to perform the action.
        """
        closest = (num_samples // every) * every
        return num_samples - closest < step_size

    def train(conf: TrainConfig, gpus, nodes=1, mode: str = 'train'):
        """
        Main training/testing function.

        Args:
            conf (TrainConfig): Configuration object.
            gpus (list): List of GPU IDs.
            nodes (int): Number of nodes.
            mode (str): 'train' or 'test'.
        """
        print(f'Configuration Name: {conf.name}')
        model = LitModel(conf)

        os.makedirs(conf.logdir, exist_ok=True)
        checkpoint = ModelCheckpoint(
            dirpath=conf.logdir,
            save_last=True,
            save_top_k=1,
            every_n_train_steps=conf.save_every_samples // conf.batch_size_effective
        )
        checkpoint_path = os.path.join(conf.logdir, 'last.ckpt')
        print(f'Checkpoint path: {checkpoint_path}')

        if os.path.exists(checkpoint_path):
            resume = checkpoint_path
            print('Resuming from the latest checkpoint...')
        else:
            resume = conf.continue_from.path if conf.continue_from is not None else None
            if resume:
                print(f'Resuming from checkpoint: {resume}')

        tb_logger = pl_loggers.TensorBoardLogger(save_dir=conf.logdir, name='', version='')

        # Initialize plugins for distributed training
        plugins = []
        accelerator = None
        if len(gpus) > 1 or nodes > 1:
            accelerator = 'ddp'
            from pytorch_lightning.plugins import DDPPlugin
            plugins.append(DDPPlugin(find_unused_parameters=False))

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
        elif mode == 'test':
            # Load the latest checkpoint for testing
            dummy = DataLoader(TensorDataset(torch.tensor([0.] * conf.batch_size)), batch_size=conf.batch_size)
            eval_path = conf.eval_path or checkpoint_path
            print(f'Loading evaluation from: {eval_path}')
            state = torch.load(eval_path, map_location='cpu')
            print(f'Checkpoint step: {state["global_step"]}')
            model.load_state_dict(state['state_dict'])
            out = trainer.test(model, dataloaders=dummy)

            # Process and log outputs
            out = out[0]
            print(out)

            if get_rank() == 0:
                for k, v in out.items():
                    tb_logger.experiment.add_scalar(k, v, state['global_step'] * conf.batch_size_effective)

                # Save to evaluation file
                tgt = f'evals/{conf.name}.txt'
                os.makedirs(os.path.dirname(tgt), exist_ok=True)
                with open(tgt, 'a') as f:
                    f.write(json.dumps(out) + "\n")
        else:
            raise NotImplementedError(f"Mode '{mode}' is not supported.")

