import os
from typing import List
import math
import warnings
import logging
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, _LRScheduler


def countParam(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


def get_logger(output, name='train', log_level=1):
    log_levels = {
        0: logging.WARNING,
        1: logging.INFO,
        2: logging.DEBUG
    }
    logging.basicConfig(format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                        datefmt="%m/%d/%Y %H:%M:%S",
                        level=log_levels[log_level],
                        filename=f'{output}/{name}.log',
                        filemode='a')

    msg_log_level = 'log_level option {} is invalid. Valid options are {}.'.format(log_level,
                                                                                   log_levels.keys())
    assert log_level in log_levels, msg_log_level
    logger = logging.getLogger(__name__)
    chlr = logging.StreamHandler()  # 打印到日志的同时，依然输出到控制台
    logger.addHandler(chlr)
    return logger


def setup_seed(seed=3407):
    """
    Torch.manual_seed(3407) is all you need. refer to: https://arxiv.org/abs/2109.08203

    :param seed: 3407
    :return: None
    """
    os.environ['PYTHONHASHSEED'] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    np.random.seed(seed)
    random.seed(seed)


def augment_affine(img_in, seg_in, mind_in=None, strength=2.5):
    """
    3D affine augmentation on image and segmentation mini-batch on GPU.
    (affine transf. is centered: trilinear interpolation and zero-padding used for sampling)
    :param input: img_in batch (torch.cuda.FloatTensor), seg_in batch (torch.cuda.LongTensor)
    :return: augmented BxCxTxHxW image batch (torch.cuda.FloatTensor), augmented BxTxHxW seg batch (torch.cuda.LongTensor)
    """
    # use_what = np.random.choice([0])  # 1, 2,
    B, C, D, H, W = img_in.size()

    # if use_what == 0:
    # 仿射变形
    affine_matrix = (torch.eye(3, 4).unsqueeze(0) + torch.randn(B, 3, 4) * strength).to(img_in.device)

    # elif use_what == 1:
    #     # 缩放
    #     z = np.random.choice([0.8, 0.9, 1.1, 1.2])
    #     affine_matrix = torch.tensor([[z, 0, 0, 0],
    #                                   [0, z, 0, 0],
    #                                   [0, 0, z, 0]], dtype=torch.float32).to(img_in.device)
    #
    # elif use_what == 2:
    #     # 旋转
    #     angle = np.random.choice([-10, -5, 5, 10]) * math.pi / 180
    #     affine_matrix = torch.tensor([[math.cos(angle), math.sin(-angle), math.sin(-angle), 0],
    #                                   [math.sin(angle), math.cos(angle), math.sin(-angle), 0],
    #                                   [math.sin(angle), math.sin(angle), math.cos(angle), 0]],
    #                                  dtype=torch.float32).to(img_in.device)

    meshgrid = F.affine_grid(affine_matrix.expand(B, 3, 4), size=[B, 1, D, H, W], align_corners=False)

    img_out = F.grid_sample(img_in, meshgrid, padding_mode='border', align_corners=False)
    seg_out = F.grid_sample(seg_in.float(), meshgrid, mode='nearest', align_corners=False).long()
    if mind_in:
        mind_out = F.grid_sample(mind_in, meshgrid, padding_mode='border', align_corners=False)
        return img_out, seg_out, mind_out  # .type(dtype=torch.uint8) 保存成 nii 需要 uint8类型
    else:
        return img_out, seg_out


class LinearWarmupCosineAnnealingLR(_LRScheduler):

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int,
        max_epochs: int,
        warmup_start_lr: float = 0.0,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        """
        Args:
            optimizer (Optimizer): Wrapped optimizer.
            warmup_epochs (int): Maximum number of iterations for linear warmup
            max_epochs (int): Maximum number of iterations
            warmup_start_lr (float): Learning rate to start the linear warmup. Default: 0.
            eta_min (float): Minimum learning rate. Default: 0.
            last_epoch (int): The index of last epoch. Default: -1.
        """
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min

        super(LinearWarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """
        Compute learning rate using chainable form of the scheduler
        """
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.",
                UserWarning,
            )

        if self.last_epoch == 0:
            return [self.warmup_start_lr] * len(self.base_lrs)
        elif self.last_epoch < self.warmup_epochs:
            return [
                group["lr"] + (base_lr - self.warmup_start_lr) / (self.warmup_epochs - 1)
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]
        elif self.last_epoch == self.warmup_epochs:
            return self.base_lrs
        elif (self.last_epoch - 1 - self.max_epochs) % (2 * (self.max_epochs - self.warmup_epochs)) == 0:
            return [
                group["lr"] + (base_lr - self.eta_min) *
                (1 - math.cos(math.pi / (self.max_epochs - self.warmup_epochs))) / 2
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]

        return [
            (1 + math.cos(math.pi * (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs))) /
            (
                1 +
                math.cos(math.pi * (self.last_epoch - self.warmup_epochs - 1) / (self.max_epochs - self.warmup_epochs))
            ) * (group["lr"] - self.eta_min) + self.eta_min for group in self.optimizer.param_groups
        ]

    def _get_closed_form_lr(self) -> List[float]:
        """
        Called when epoch is passed as a param to the `step` function of the scheduler.
        """
        if self.last_epoch < self.warmup_epochs:
            return [
                self.warmup_start_lr + self.last_epoch * (base_lr - self.warmup_start_lr) / (self.warmup_epochs - 1)
                for base_lr in self.base_lrs
            ]

        return [
            self.eta_min + 0.5 * (base_lr - self.eta_min) *
            (1 + math.cos(math.pi * (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)))
            for base_lr in self.base_lrs
        ]


class ImgTransform:
    """
    Image intensity normalization.

    Params:
        :scale_type: normalization way, default mean-std scaled
        :img: ndarray or tensor
    Return:
        scaled img
    """
    def __init__(self, scale_type="mean-std"):
        assert scale_type in ["mean-std", "max-min", "old-way", None], \
            f"scale type include ['mean-std', 'max-min', 'old-way', 'None'], but got {scale_type}"
        self.scale_type = scale_type

    def __call__(self, img):
        if self.scale_type == "mean-std":
            return (img - img.mean()) / img.std()
        if self.scale_type == "max-min":
            return (img - img.min()) / (img.max() - img.min())
        if self.scale_type == "old-way":
            return img / 1024.0 + 1.0
        if self.scale_type is None:
            return img
