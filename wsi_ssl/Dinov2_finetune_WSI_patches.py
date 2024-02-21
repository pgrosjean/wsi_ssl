# importing base python dependencies
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from datetime import datetime, date
import os
import random
from pathlib import Path
import copy
from tqdm import tqdm
import multiprocessing

# importing torch related dependencies
import torch
import torch.nn as nn
import torch.distributed as dist
from torch import optim
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
import torchvision.transforms as transforms
from torchvision.transforms import functional as TF
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

# importing the DINO projection head implementation from Lightly
from lightly.models.modules.heads import DINOProjectionHead

# importing histomicstok for pathology specific augmentation
from histomicstk.preprocessing.augmentation import rgb_perturb_stain_concentration

# importing other dependencies
import pandas as pd
import zarr
import numpy as np

# importing logging dependencies
import wandb
from pytorch_lightning.loggers import WandbLogger


###########
###########
###########
###########
# Defining the Data Utilities
###########
###########
###########
###########


class RGBPerturbStainConcentrationTransform():
    def __init__(self, **kwargs):
        """
        Initializes the transform with parameters for rgb_perturb_stain_concentration.

        Parameters
        ----------
        kwargs : dict
            Arguments passed to the rgb_perturb_stain_concentration function.
        """
        self.kwargs = kwargs

    def __call__(self, img):
        """
        Apply the stain perturbation transform to the input image.

        Parameters
        ----------
        img : torch.Tensor
            The input image.

        Returns
        -------
        torch.Tensor
            The transformed image as a PyTorch tensor.
        """
        # Ensure img is a NumPy array
        img_np = img.permute(1, 2, 0).numpy()

        # Apply the HistomicsTK stain concentration perturbation
        img_np = rgb_perturb_stain_concentration(img_np, **self.kwargs)

        # Convert back to PIL Image, then to Tensor if necessary
        img_tensor = torch.Tensor(img_np).permute(2, 0, 1)

        return img_tensor


def collate_zarr_files(directory: str):
    # List to store the paths of .tif files
    files = []
    # Walk through directory recursively
    for dir, _, filenames in os.walk(directory):
        if str(dir.split('/')[-1]).endswith('.zarr'):
            files.append(dir)
    return np.array(files)


class WSIPathSSLDataset(Dataset):
    def __init__(self, base_directory):
        self.zarr_files = self._get_zarr_files(base_directory)
        self.length, self.intervals = self._get_dataset_len()
        self.zf = np.repeat(self.zarr_files, self.intervals)
        self.inner_idx = np.hstack([np.arange(x) for x in self.intervals])
        self.resize_transform = transforms.Resize((224, 224), antialias=True)

    def _get_dataset_len(self):
        total_len = []
        print('Collating all files...')
        zarr_files = []
        for f in tqdm(self.zarr_files):
            try:
                root = zarr.open(f, 'r')
                num_tiles = np.array(root['arr_1']).shape[0]
                total_len.append(num_tiles)
                zarr_files.append(f)
            except:
                print(f'error reading in {f}')
        self.zarr_files = np.array(zarr_files)
        total_len = np.array(total_len)
        dataset_len = np.sum(total_len)
        return int(dataset_len), total_len
    
    def _get_zarr_files(self, base_dir):
        return collate_zarr_files(base_dir)

    def _random_transform(self, image):
        # Define a list of PyTorch transform functions
        transform_functions = [
            transforms.RandomHorizontalFlip(p=1),
            transforms.RandomRotation([-90, 90]),
            transforms.RandomVerticalFlip(p=1),
            RGBPerturbStainConcentrationTransform(sigma1=0.7, sigma2=0.7)
        ]
        
        # Apply the transform function to the image
        n_transforms = random.choice([1, 2, 3])
        for _ in range(n_transforms):
            try:
                transform_function = random.choice(transform_functions)
                image = transform_function(image)
            except:
                pass
        # Return the transformed images as a tensor
        transformed_image = image
        return transformed_image

    def __getitem__(self, idx):
        zf = self.zf[idx]
        ii = self.inner_idx[idx]
        root = zarr.open(zf, 'r')
        im = torch.Tensor(np.array(root['arr_0'][ii])).permute(2, 0, 1)
        t_im = self._random_transform(im)
        im = self.resize_transform(im)/255
        t_im = self.resize_transform(t_im)/255
        return im, t_im

    def __len__(self):
        return int(self.length)


###########
###########
###########
###########
# Defining the model for fine-tuning with DINO V1 Loss
###########
###########
###########
###########


import numpy as np
import warnings
from typing import Optional

def cosine_schedule(
    step: int,
    max_steps: int,
    start_value: float,
    end_value: float,
    period: Optional[int] = None
) -> float:
    """Gradually modify start_value to end_value using cosine decay.

    Args:
        step: Current step number.
        max_steps: Total number of steps.
        start_value: Starting value.
        end_value: Target value.
        period: Number of steps for a full cosine cycle, defaults to max_steps.

    Returns:
        Cosine decay value.
    """
    if step < 0:
        raise ValueError("step must be non-negative")
    if max_steps < 1:
        raise ValueError("max_steps must be at least 1")
    if period is not None and period <= 0:
        raise ValueError("period must be positive")
    
    # Use max_steps as period if period is None or enforce end_value at the last step
    effective_period = period if period is not None else max_steps
    
    # Special handling to avoid potential division by zero and ensure correct final value
    if step >= max_steps:
        return end_value
    
    decay = end_value + 0.5 * (start_value - end_value) * (1 + np.cos(np.pi * step / effective_period))
    return decay


class DINOLossSingleViews(nn.Module):
    def __init__(
        self,
        output_dim: int = 65536,
        warmup_teacher_temp: float = 0.04,
        teacher_temp: float = 0.07,
        warmup_teacher_temp_epochs: int = 10,
        student_temp: float = 0.1,
        center_momentum: float = 0.9,
    ):
        super().__init__()
        self.warmup_teacher_temp_epochs = warmup_teacher_temp_epochs
        self.teacher_temp = teacher_temp
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, output_dim))
        self.teacher_temp_schedule = torch.linspace(
            start=warmup_teacher_temp,
            end=teacher_temp,
            steps=warmup_teacher_temp_epochs,
        )

    def forward(
        self,
        teacher_out: torch.Tensor,
        student_out: torch.Tensor,
        epoch: int,
        validation: bool = False,
    ) -> torch.Tensor:
        """Cross-entropy between softmax outputs of the teacher and student
        networks.

        Paramters
        ---------
            teacher_out:
                feature tensors from the teacher model. Each tensor is assumed 
                to contain features from one view of the batch and have length batch_size.
            student_out:
                feature tensors from the student model. Each tensor is assumed 
                to contain features from one view of the batch and have length batch_size.
            epoch:
                The current training epoch.

        Returns
        -------
            The average cross-entropy loss.

        """
        # get teacher temperature
        if epoch < self.warmup_teacher_temp_epochs:
            teacher_temp = self.teacher_temp_schedule[epoch]
        else:
            teacher_temp = self.teacher_temp

        t_out = F.softmax((teacher_out - self.center) / teacher_temp, dim=-1)
        s_out = F.log_softmax(student_out / self.student_temp, dim=-1)
        t_out = t_out.unsqueeze(0)
        s_out = s_out.unsqueeze(0)
        
        # calculate feature similarities where:
        # b -> batch_size, t -> n_views_teacher, s -> n_views_student, d -> output_dim
        loss = -torch.einsum("tbd,sbd->ts", t_out, s_out).squeeze()
        loss = loss/teacher_out.shape[0]
        if not validation:
            self.update_center(teacher_out)
        return loss

    @torch.no_grad()
    def update_center(self, teacher_out: torch.Tensor) -> None:
        """Moving average update of the center used for the teacher output.

        Args:
            teacher_out:
                Stacked output from the teacher model.

        """
        batch_center = torch.mean(teacher_out, dim=0, keepdim=True)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(batch_center)
            batch_center = batch_center / dist.get_world_size()

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (
            1 - self.center_momentum)


class DINOViT(pl.LightningModule):
    def __init__(self,
                 lr=1e-3,
                 max_epoch_number=500,
                 num_register_tokens=4,
                 num_patches=256,
                 proj_dim=2048):
        super().__init__()
        self.lr = lr
        self.max_epoch_number = max_epoch_number
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, 768))
        assert num_register_tokens >= 0
        dinov2_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg').cuda()
        self.cls_token = dinov2_model.cls_token
        self.register_tokens = dinov2_model.register_tokens
        self.patch_embed = dinov2_model.patch_embed
        self.student_backbone = nn.Sequential(*dinov2_model.blocks)
        self.student_head = DINOProjectionHead(768, 2048, 256, proj_dim, batch_norm=False)
        self.teacher_backbone = copy.deepcopy(self.student_backbone)
        self.teacher_head = DINOProjectionHead(768, 2048, 256, proj_dim, batch_norm=False)
        # Making the teacher model require no
        for param in self.teacher_backbone.parameters():
            param.requires_grad = False
        for param in self.teacher_head.parameters():
            param.requires_grad = False
        self.dino_loss = DINOLossSingleViews(output_dim=proj_dim, warmup_teacher_temp_epochs=5)
        
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("DINOViT")
        parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for training.")
        parser.add_argument("--max_epoch_number", type=int, default=500, help="Maximum number of epochs.")
        parser.add_argument("--proj_dim", type=int, default=1024, help="The embedding dimension from the DINO head.")
        parser.add_argument("--num_register_tokens", type=int, default=4, help="Number of register tokens.")
        parser.add_argument("--num_patches", type=int, default=256, help="The number of patch tokens for the ViT.")
        return parent_parser
    
    def forward(self, X):
        X = self.prepare_tokens_with_masks(X)
        h = self.student_backbone(X)
        # Extracting out the cls token
        h = h[:, 0]
        z = self.student_head(h)
        return z

    def forward_teacher(self, X):
        X = self.prepare_tokens_with_masks(X)
        h = self.teacher_backbone(X)
        # Extracting out the cls token
        h = h[:, 0]
        z = self.teacher_head(h)
        return z

    def prepare_tokens_with_masks(self, x, masks=None):
        """
        This function is adapted from DINOV2 from meta.
        https://github.com/facebookresearch/dinov2/blob/main/dinov2/models/vision_transformer.py
        """
        B, nc, w, h = x.shape
        x = self.patch_embed(x)
        x = x + self.pos_embed.repeat(B, 1, 1)
        if masks is not None:
            x = torch.where(masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x)
        
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)

        if self.register_tokens is not None:
            x = torch.cat(
                (
                    x[:, :1],
                    self.register_tokens.expand(x.shape[0], -1, -1),
                    x[:, 1:],
                ),
                dim=1,
            )
        return x

    @torch.no_grad()
    def update_momentum(self, model: nn.Module, model_ema: nn.Module, m: float):
        """
        Updates model_ema with Exponential Moving Average of model
        """
        for model_ema, model in zip(model_ema.parameters(), model.parameters()):
            model_ema.data = model_ema.data * m + model.data * (1.0 - m)
    
    @torch.no_grad()
    def _calculate_nuc_norm(self, embeddings):
        embeddings = embeddings.to(torch.float)
        _, S, _ = torch.linalg.svd(embeddings)
        nuc_norm = S.sum()
        nuc_norm = -1 * nuc_norm
        return nuc_norm
    
    def training_step(self, batch, _):
        X, X_t = batch
        momentum = cosine_schedule(self.current_epoch, 10, 0.996, 1)
        # EMA update of backbone and head
        self.update_momentum(self.student_backbone, self.teacher_backbone, m=momentum)
        self.update_momentum(self.student_head, self.teacher_head, m=momentum)
        # Forward Pass through teacher
        teacher_out = self.forward_teacher(X_t)
        # Detaching teacher output for stop gradient
        teacher_out = teacher_out.detach()
        # Forward Pass for student
        student_out = self.forward(X)
        # Calculating the loss
        loss_dino = self.dino_loss(teacher_out, student_out, epoch=self.current_epoch)
        loss = loss_dino
        # Calculating the negative nuclear norm to asses representational collapse
        neg_nuclear_norm = self._calculate_nuc_norm(torch.vstack([student_out, teacher_out]))
        loss_dict = {'train_loss': loss, 'train_nuc_norm': neg_nuclear_norm}
        self.log_dict(loss_dict, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, _):
        X, X_t = batch
        # Forward Pass through teacher
        teacher_out = self.forward_teacher(X_t)
        # Detaching teacher output for stop gradient
        teacher_out = teacher_out.detach()
        # Forward Pass for student
        student_out = self.forward(X)
        # Calculating the loss
        loss_dino = self.dino_loss(teacher_out, student_out, epoch=self.current_epoch)
        loss = loss_dino
        # Calculating the negative nuclear norm to asses representational collapse
        neg_nuclear_norm = self._calculate_nuc_norm(torch.vstack([student_out, teacher_out]))
        loss_dict = {'val_loss': loss, 'val_nuc_norm': neg_nuclear_norm}
        self.log_dict(loss_dict, on_step=True, on_epoch=True)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr)
        def warmup_fn(epoch):
            warmup_epochs = 500
            if epoch < warmup_epochs:
                return (epoch / warmup_epochs)*self.lr
            elif epoch == 0:
                return (0.5 / warmup_epochs)*self.lr
            else:
                return self.lr
        # Adding warmup scheduler
        warmup_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_fn)
        # After warmup, use a scheduler of your choice
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.max_epoch_number)
        return [optimizer], [{"scheduler": warmup_scheduler, "interval": "step"}, {"scheduler": scheduler, "interval": "epoch"}]


################
################
################
################
# Defining Main
################
################
################
################

def main():
    # Seeding everything to ensure reproducibility
    pl.seed_everything(1)
    
    # Argparsing
    desc = "Script for Training JUMP Context ViT."
    parser = ArgumentParser(
        description=desc, formatter_class=ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--gpu_num", type=int, default=3, help="GPU number to use.")
    parser.add_argument("--name", type=str, help="Name of experiment for logger.")
    parser.add_argument("--logs", type=Path, help="Path to model logs and checkpoints.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch Size (note: actual batch is batch_size * num_sc_ims.")
    parser = DINOViT.add_model_specific_args(parser)
    args = parser.parse_args()
    dict_args = vars(args)

    
    # ------------
    # data
    # ------------
    print("Generating Training and Validation DataSets...")
    train_zarr_path = '/scratch/pgrosjean/path_ssl/prostate_dataset/train/'
    val_zarr_path = '/scratch/pgrosjean/path_ssl/prostate_dataset/validation/'
    train_dataset = WSIPathSSLDataset(train_zarr_path)
    val_dataset = WSIPathSSLDataset(val_zarr_path)
    train_dl = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size, num_workers=12)
    val_dl = DataLoader(val_dataset, shuffle=False, batch_size=args.batch_size, num_workers=0)
    
    # ------------
    # model
    # ------------
    print("Instantiating Model...")
    model = DINOViT(lr=args.lr,
                    max_epoch_number=args.max_epoch_number,
                    num_register_tokens=args.num_register_tokens,
                    num_patches=args.num_patches,
                    proj_dim=args.proj_dim)

    # ------------
    # training
    # ------------
    # Initializing wandb logger
    print("Initializing WandB Logger...")
    logger = WandbLogger(save_dir=args.logs,
                         name=f'{args.name}',
                         project="Prostate_Path_SSL"
    )
    # Defining Callbacks
    early_stop = EarlyStopping(
        monitor="val_loss_epoch", min_delta=1e-5, patience=30, verbose=False, mode="min"
    )
    checkpoint_callback = ModelCheckpoint(
        monitor="train_loss", mode="min", save_last=True, every_n_train_steps=200, save_top_k=2,
    )

    # Creating Trainer from argparse args
    if args.gpu_num == -1:
        gpu_num = -1
    else:
        gpu_num = [args.gpu_num]
        
    trainer = pl.Trainer(accelerator="gpu",
                         devices=gpu_num,
                         callbacks=[early_stop, checkpoint_callback],
                         default_root_dir=args.logs,
                         logger=logger,
                         enable_checkpointing=True,
                         profiler="simple",
                         max_epochs=int(args.max_epoch_number),
                         log_every_n_steps=10,
                         precision=16)
    # Training the model
    print("Training Model...")
    trainer.fit(model, train_dl, val_dl)

    
if __name__ == "__main__":
    main()









