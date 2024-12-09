import copy
import json
import os
import re
from contextlib import contextmanager

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from torch import nn
from torch.cuda import amp
from torch.optim.optimizer import Optimizer
from torch.utils.data.dataset import TensorDataset
from torchvision.utils import make_grid, save_image

from config import TrainConfig
from dataset import EEGDataset, ConditionalEEGImageDataset, data_paths
from dist_utils import get_world_size, get_rank
from lmdb_writer import LMDBImageWriter  # Ensure this is implemented
from metrics import evaluate_fid, evaluate_lpips  # Ensure these are implemented
from renderer import render_condition, render_uncondition  # Ensure these are implemented
from choices import TrainMode, ModelType, ModelName, LossType


class LitModel(pl.LightningModule):
    def __init__(self, conf: TrainConfig):
        super().__init__()
        assert conf.train_mode != TrainMode.manipulate, "Manipulate mode not supported in this setup."
        
        if conf.seed is not None:
            pl.seed_everything(conf.seed)

        self.save_hyperparameters(conf.as_dict_jsonable())
        self.conf = conf

        # Initialize the model based on configuration
        self.model = conf.make_model_conf().make_model()
        self.ema_model = copy.deepcopy(self.model)
        self.ema_model.requires_grad_(False)
        self.ema_model.eval()

        # Calculate and print model size
        model_size = sum(p.numel() for p in self.model.parameters())
        print(f'Model parameters: {model_size / 1e6:.2f}M')

        # Initialize samplers
        self.sampler = conf.make_diffusion_conf().make_sampler()
        self.eval_sampler = conf.make_eval_diffusion_conf().make_sampler()

        # Samplers for latent diffusion if required
        if conf.train_mode.use_latent_net():
            self.latent_sampler = conf.make_latent_diffusion_conf().make_sampler()
            self.eval_latent_sampler = conf.make_latent_eval_diffusion_conf().make_sampler()
        else:
            self.latent_sampler = None
            self.eval_latent_sampler = None

        # Register a buffer for consistent sampling
        self.register_buffer('x_T', torch.randn(conf.sample_size, 3, conf.img_size, conf.img_size))

        # Load pre-trained model if specified
        if conf.pretrain is not None:
            print(f'Loading pre-trained model from {conf.pretrain.name}...')
            state = torch.load(conf.pretrain.path, map_location='cpu')
            print(f'Pre-trained model step: {state["global_step"]}')
            self.load_state_dict(state['state_dict'], strict=False)

        # Load latent statistics if provided
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
        assert self.conf.model_type.has_autoenc(), "ModelType does not include an encoder."
        cond = self.ema_model.encoder(x)
        return cond

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
        with amp.autocast(False):
            model = self.ema_model if ema_model else self.model
            gen = self.eval_sampler.sample(model=model, noise=noise, x_start=x_start)
            return gen

    def setup(self, stage=None) -> None:
        """
        Initialize datasets and set seeds for each worker.
        """
        if self.conf.seed is not None:
            seed = self.conf.seed * get_world_size() + get_rank()
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            print(f'Set seed: {seed}')

        self.train_data = self.conf.make_dataset()
        print(f'Training data size: {len(self.train_data)}')
        self.val_data = self.train_data  # For simplicity; adjust as needed
        print(f'Validation data size: {len(self.val_data)}')

    def _train_dataloader(self, drop_last=True):
        """
        Create the actual DataLoader for training.
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
        Return the appropriate DataLoader based on TrainMode.
        """
        print('Initializing training DataLoader...')
        if self.conf.train_mode.require_dataset_infer():
            if self.conds is None:
                self.conds = self.infer_whole_dataset()
                self.conds_mean.data = self.conds.float().mean(dim=0, keepdim=True)
                self.conds_std.data = self.conds.float().std(dim=0, keepdim=True)
            print(f'Mean: {self.conds_mean.mean()}, Std: {self.conds_std.mean()}')

            conf = self.conf.clone()
            conf.batch_size = self.batch_size
            data = TensorDataset(self.conds)
            return conf.make_loader(data, shuffle=True)
        else:
            return self._train_dataloader()

    @property
    def batch_size(self):
        """
        Calculate the local batch size for each worker.
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
        Optionally render images from latents.
        """
        data = self.conf.make_dataset()
        if isinstance(data, CelebAlmdb) and data.crop_d2c:
            data.transform = make_transform(self.conf.img_size, flip_prob=0, crop_d2c=True)
        else:
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
                writer = contextmanager(lambda: None)()
        else:
            writer = contextmanager(lambda: None)()

        with writer:
            for batch in tqdm(loader, total=len(loader), desc='Infer Latents'):
                with torch.no_grad():
                    cond = model.encoder(batch['img'].to(self.device))
                    idx = batch['index']
                    idx = self.all_gather(idx)
                    if idx.dim() == 2:
                        idx = idx.flatten(0, 1)
                    argsort = idx.argsort()

                    if with_render:
                        noise = torch.randn(len(cond), 3, self.conf.img_size, self.conf.img_size, device=self.device)
                        render = sampler.sample(
                            model=self.model,
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
        Define the training step.
        """
        with amp.autocast(False):
            if self.conf.train_mode.require_dataset_infer():
                cond = batch[0]
                if getattr(self.conf, 'latent_znormalize', False):
                    cond = self.normalize(cond)
            else:
                imgs, idxs = batch['img'], batch['index']
                x_start = imgs

            if self.conf.train_mode == TrainMode.diffusion:
                # Standard diffusion training
                t, weight = self.T_sampler.sample(len(x_start), x_start.device)
                losses = self.sampler.training_losses(model=self.model, x_start=x_start, t=t)
            elif self.conf.train_mode.is_latent_diffusion():
                # Latent diffusion training
                t, weight = self.T_sampler.sample(len(cond), cond.device)
                latent_losses = self.latent_sampler.training_losses(model=self.model.latent_net, x_start=cond, t=t)
                losses = {
                    'latent': latent_losses['loss'],
                    'loss': latent_losses['loss']
                }
            else:
                raise NotImplementedError(f"Unsupported TrainMode: {self.conf.train_mode}")

            loss = losses['loss'].mean()
            # Aggregate losses across GPUs
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
        Actions to perform at the end of each training batch.
        """
        if self.is_last_accum(batch_idx):
            if self.conf.train_mode == TrainMode.latent_diffusion:
                # Update EMA for latent network
                ema(self.model.latent_net, self.ema_model.latent_net, self.conf.ema_decay)
            else:
                # Update EMA for the entire model
                ema(self.model, self.ema_model, self.conf.ema_decay)

            # Log samples and evaluate scores
            if self.conf.train_mode.require_dataset_infer():
                imgs = None
            else:
                imgs = batch['img']
            self.log_sample(x_start=imgs)
            self.evaluate_scores()

    def on_before_optimizer_step(self, optimizer: Optimizer, optimizer_idx: int) -> None:
        """
        Clip gradients before the optimizer step to prevent exploding gradients.
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
                        # Generate images from latents
                        gen = render_uncondition(
                            conf=self.conf,
                            model=model,
                            x_T=x_T,
                            sampler=self.eval_sampler,
                            latent_sampler=self.eval_latent_sampler,
                            conds_mean=self.conds_mean,
                            conds_std=self.conds_std
                        )
                    else:
                        if not use_xstart and self.conf.model_type.has_autoenc():
                            cond = torch.randn(len(x_T), self.conf.style_ch, device=self.device)
                            cond = model.noise_to_cond(cond)
                        else:
                            if interpolate:
                                with amp.autocast(self.conf.fp16):
                                    cond = model.encoder(_xstart)
                                    i = torch.randperm(len(cond))
                                    cond = (cond + cond[i]) / 2
                            else:
                                cond = None
                        gen = self.eval_sampler.sample(model=model, noise=x_T, cond=cond, x_start=_xstart)
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
                if self.conf.model_type.has_autoenc() and self.conf.model_type.can_sample():
                    do(self.model, '', use_xstart=False)
                    do(self.ema_model, '_ema', use_xstart=False)
                    do(self.model, '_enc', use_xstart=True, save_real=True)
                    do(self.ema_model, '_enc_ema', use_xstart=True, save_real=True)
                elif self.conf.train_mode.use_latent_net():
                    do(self.model, '', use_xstart=False)
                    do(self.ema_model, '_ema', use_xstart=False)
                    do(self.model, '_enc', use_xstart=True, save_real=True)
                    do(self.model, '_enc_nodiff', use_xstart=True, save_real=True, no_latent_diff=True)
                    do(self.ema_model, '_enc_ema', use_xstart=True, save_real=True)
                else:
                    do(self.model, '', use_xstart=True, save_real=True)
                    do(self.ema_model, '_ema', use_xstart=True, save_real=True)

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

        def lpips(model, postfix):
            if self.conf.model_type.has_autoenc() and self.conf.train_mode.is_autoenc():
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

        if self.conf.eval_every_samples > 0 and self.num_samples > 0 and is_time(self.num_samples, self.conf.eval_every_samples, self.conf.batch_size_effective):
            print(f'Evaluating FID at {self.num_samples} samples...')
            lpips(self.model, '')
            fid(self.model, '')

        if self.conf.eval_ema_every_samples > 0 and self.num_samples > 0 and is_time(self.num_samples, self.conf.eval_ema_every_samples, self.conf.batch_size_effective):
            print(f'Evaluating FID for EMA model at {self.num_samples} samples...')
            fid(self.ema_model, '_ema')

    def configure_optimizers(self):
        """
        Configure optimizers and learning rate schedulers.
        """
        optimizer = None
        if self.conf.optimizer == 'adam':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.conf.lr, weight_decay=self.conf.weight_decay)
        elif self.conf.optimizer == 'adamw':
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.conf.lr, weight_decay=self.conf.weight_decay)
        else:
            raise NotImplementedError(f"Unsupported optimizer type: {self.conf.optimizer}")

        schedulers = []
        if self.conf.warmup > 0:
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=WarmupLR(self.conf.warmup))
            schedulers.append({'scheduler': scheduler, 'interval': 'step'})

        return {'optimizer': optimizer, 'lr_scheduler': schedulers}

    def split_tensor(self, x):
        """
        Split the tensor based on the current rank in multi-GPU setup.
        """
        n = len(x)
        rank = get_rank()
        world_size = get_world_size()
        per_rank = n // world_size
        return x[rank * per_rank:(rank + 1) * per_rank]

    def test_step(self, batch, *args, **kwargs):
        """
        Perform evaluation based on `conf.eval_programs`.
        """
        self.setup()

        print(f'Global step: {self.global_step}')
        if 'infer' in self.conf.eval_programs:
            print('Performing latent inference...')
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
                if m:
                    T = int(m.group(1))
                    print(f'Performing inference and rendering at T={T}...')
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
                if m:
                    T = int(m.group(1))
                    T_latent = int(m.group(2))
                    print(f'Evaluating FID at T={T} and T_latent={T_latent}...')
                else:
                    m = re.match(r'fidclip\(([0-9]+),([0-9]+)\)', each)
                    if m:
                        T = int(m.group(1))
                        T_latent = int(m.group(2))
                        clip_latent_noise = True
                        print(f'Evaluating FID with clipped latent noise at T={T} and T_latent={T_latent}...')
                    else:
                        _, T = each.split('fid')
                        T = int(T)
                        T_latent = None
                        print(f'Evaluating FID at T={T}...')

                self.train_dataloader()
                sampler = self.conf._make_diffusion_conf(T=T).make_sampler()
                latent_sampler = self.conf._make_latent_diffusion_conf(T_latent).make_sampler() if T_latent else None

                conf = self.conf.clone()
                conf.eval_num_images = 50_000
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

        for each in self.conf.eval_programs:
            if each.startswith('recon'):
                _, T = each.split('recon')
                T = int(T)
                print(f'Evaluating reconstruction at T={T}...')
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

        for each in self.conf.eval_programs:
            if each.startswith('inv'):
                _, T = each.split('inv')
                T = int(T)
                print(f'Evaluating reconstruction with noise inversion at T={T}...')
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
        Update the EMA (Exponential Moving Average) of the model parameters.
        """
        source_dict = source.state_dict()
        target_dict = target.state_dict()
        for key in source_dict.keys():
            target_dict[key].data.copy_(target_dict[key].data * decay + source_dict[key].data * (1 - decay))

    class WarmupLR:
        def __init__(self, warmup) -> None:
            self.warmup = warmup

        def __call__(self, step):
            return min(step, self.warmup) / self.warmup

    def is_time(num_samples, every, step_size):
        """
        Determine if it's time to perform an action based on the number of samples processed.
        """
        closest = (num_samples // every) * every
        return num_samples - closest < step_size

    def train(conf: TrainConfig, gpus, nodes=1, mode: str = 'train'):
        """
        Orchestrate the training and evaluation process.
        """
        print(f'Configuration: {conf.name}')
        model = LitModel(conf)

        os.makedirs(conf.logdir, exist_ok=True)
        checkpoint_callback = ModelCheckpoint(
            dirpath=conf.logdir,
            save_last=True,
            save_top_k=1,
            every_n_train_steps=conf.save_every_samples // conf.batch_size_effective
        )
        checkpoint_path = os.path.join(conf.logdir, 'last.ckpt')
        print(f'Checkpoint path: {checkpoint_path}')
        if os.path.exists(checkpoint_path):
            resume = checkpoint_path
            print('Resuming from the last checkpoint...')
        else:
            resume = conf.continue_from.path if conf.continue_from is not None else None

        tb_logger = pl_loggers.TensorBoardLogger(save_dir=conf.logdir, name=None, version='')

        # Configure plugins for distributed training
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
            callbacks=[checkpoint_callback, LearningRateMonitor(logging_interval='step')],
            logger=tb_logger,
            accumulate_grad_batches=conf.accum_batches,
            plugins=plugins,
            replace_sampler_ddp=True,
        )

        if mode == 'train':
            trainer.fit(model)
        elif mode == 'eval':
            # Load the latest checkpoint for evaluation
            dummy_loader = DataLoader(TensorDataset(torch.zeros(conf.batch_size)), batch_size=conf.batch_size)
            eval_path = conf.eval_path or checkpoint_path
            print(f'Loading checkpoint from: {eval_path}')
            state = torch.load(eval_path, map_location='cpu')
            print(f'Checkpoint step: {state["global_step"]}')
            model.load_state_dict(state['state_dict'])
            trainer.test(model, dataloaders=dummy_loader)

            if get_rank() == 0:
                # Log evaluation results
                for k, v in trainer.callback_metrics.items():
                    tb_logger.experiment.add_scalar(k, v, state['global_step'] * conf.batch_size_effective)

                # Save evaluation results to a file
                eval_file = f'evals/{conf.name}.txt'
                os.makedirs(os.path.dirname(eval_file), exist_ok=True)
                with open(eval_file, 'a') as f:
                    metrics = {k: v.item() for k, v in trainer.callback_metrics.items()}
                    metrics['num_samples'] = state['global_step'] * conf.batch_size_effective
                    f.write(json.dumps(metrics) + "\n")
        else:
            raise NotImplementedError(f"Unsupported mode: {mode}")


# Utility functions
def ema(source, target, decay):
    """
    Update the EMA (Exponential Moving Average) of the model parameters.
    """
    source_dict = source.state_dict()
    target_dict = target.state_dict()
    for key in source_dict.keys():
        target_dict[key].data.copy_(target_dict[key].data * decay + source_dict[key].data * (1 - decay))


class WarmupLR:
    def __init__(self, warmup) -> None:
        self.warmup = warmup

    def __call__(self, step):
        return min(step, self.warmup) / self.warmup


def is_time(num_samples, every, step_size):
    """
    Determine if it's time to perform an action based on the number of samples processed.
    """
    closest = (num_samples // every) * every
    return num_samples - closest < step_size
