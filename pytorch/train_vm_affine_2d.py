import os
import time
import argparse
import pathlib
import numpy as np
import visdom

import torch
import torch.nn as nn
from torch import optim

from utils import get_logger, countParam, LinearWarmupCosineAnnealingLR, ImgTransform, setup_seed
from utils.losses import NCCLoss
from datasets import mnist
from models import VxmAffineNet, VxmAffineNet_regress

setup_seed()


def main():
    # read/parse user command line input
    parser = argparse.ArgumentParser()

    # dataset args
    parser.add_argument("-output", help="filename (without extension) for output",
                        default="output/mnist-affine-GT/")
    parser.add_argument("-val_size", help="validation set size, which is divided from the training set",
                        type=int, default=512)
    parser.add_argument("-choose_label", help="Which number to choose for registration training",
                        type=int, default=5)

    # training args
    parser.add_argument("-choose_net", help="Choose VM backbone", type=str,
                        choices=['VxmAffineNet', 'VxmAffineNet_regress'],
                        default='VxmAffineNet_regress')
    parser.add_argument("-batch_size", help="Dataloader batch size", type=int, default=64)
    parser.add_argument("-choose_optim", help="Choose optimizer", type=str, choices=['Adam', 'AdamW'], default='Adam')
    parser.add_argument("-lr", help="Optimizer learning rate, keep pace with batch_size",
                        type=float, default=1e-4)  # 0.005 for AdamW, 4e-4 for Adam
    parser.add_argument("-apply_lr_scheduler", help="Need lr scheduler or not", action="store_true")
    parser.add_argument("-warmup_epochs", help="epochs for Warmup scheduler", type=int, default=10)
    parser.add_argument("-epochs", help="Train epochs", type=int, default=100)  # 收敛很快
    parser.add_argument("-resume", help="Path to pretrained model to continute training",
                        default=None)  # "output/mnist/reg_net_199.pth"
    parser.add_argument("-val_interval", help="validation and saving interval", type=int, default=1)
    parser.add_argument("-save_interval", help="validation and saving interval", type=int, default=50)
    parser.add_argument("-is_visdom", help="Using Visdom to visualize Training process",
                        type=lambda s: False if s == "False" else True, default=True)
    parser.add_argument("-num_workers", help="Dataloader num_workers", type=int, default=2)

    # losses args
    parser.add_argument("-gap_size", type=int, help="you know", choices=[1, 2, 4, 6], default=4)
    parser.add_argument("-use_gap", help="If choose VxmAffineNet_regress as net, you can use gap for affine mat regressing",
                        type=lambda s: False if s == "False" else True, default=False)
    parser.add_argument("-sim_loss", type=str, help="similarity criterion", choices=['MSE', 'NCC'],
                        default='MSE')
    parser.add_argument("-alpha", type=float, help="weight for regularization loss",
                        default=0.025)  # ncc: 1.5, mse: 0.025, MIND-SSC: 4.0, VM: 0.01
    parser.add_argument("-sim_weight", help="similarity loss weight", type=float, default=1.0)

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(args.output):
        pathlib.Path(args.output).mkdir(parents=True, exist_ok=True)

    logger = get_logger(args.output)
    logger.info(f"Training with choose label {args.choose_label}, output to {args.output}")

    train_loader = mnist(for_what='train',
                         batch_size=args.batch_size,
                         val_size=args.val_size,
                         choose_label=args.choose_label,
                         num_workers=args.num_workers)

    val_loader = mnist(for_what='val',
                       batch_size=4,  # 为了展示方便
                       val_size=args.val_size,
                       choose_label=args.choose_label,
                       num_workers=args.num_workers)

    end_epoch = args.epochs  # 300

    # initialise trainable network parts
    enc_nf = [16, 32, 32, 32]
    dec_nf = [32, 32, 32, 32, 32, 16, 16]

    if args.choose_net == 'VxmAffineNet_regress':
        net = VxmAffineNet_regress
    else:
        net = VxmAffineNet
    reg_net = net(
        inshape=[32, 32],
        nb_unet_features=[enc_nf, dec_nf])
    reg_net.to(device)

    logger.info(f'{args.choose_net} reg_net params: {countParam(reg_net)}, '
                f'training with {args.choose_optim} with lr of {args.lr}')

    if args.resume:
        reg_net = reg_net.load(args.resume, device=device).to(device)
        logger.info(f"Training resume from {args.resume}")

    reg_net.train()

    # train using Adam with weight decay and exponential LR decay
    optimizer = getattr(optim, args.choose_optim)(reg_net.parameters(), lr=args.lr)
    # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999) if args.apply_lr_scheduler else None
    if args.apply_lr_scheduler:
        scheduler = LinearWarmupCosineAnnealingLR(optimizer,
                                                  warmup_epochs=args.warmup_epochs,
                                                  max_epochs=args.epochs) if args.apply_lr_scheduler else None

    # losses
    if args.sim_loss == "MSE":
        sim_criterion = nn.MSELoss()
        args.alpha = 0.025  # 根据 voxelmorph 论文的超参数实验设置
    elif args.sim_loss == "NCC":
        sim_criterion = NCCLoss()
        args.alpha = 1.5

    steps, best_acc = 0, 0
    run_loss = np.zeros([end_epoch, 3])

    if args.is_visdom:
        # 通过 visdom 实时可视化，需要提前在终端运行服务: python -m visdom.server
        vis = visdom.Visdom()
        logger.info("visdom starting, need to open the server: python -m visdom.server")
        loss_opts = {'xlabel': 'epochs',
                     'ylabel': 'loss',
                     'title': 'Loss Line',
                     'legend': ['total loss', 'sim loss', 'grad loss']}
        lr_opts = {'xlabel': 'epochs', 'ylabel': 'lr', 'title': 'LR Line'}

    for epoch in range(end_epoch):
        t0 = time.time()
        for moving_img, fixed_img in train_loader:
            steps += 1

            moving_img, fixed_img = moving_img.to(device), fixed_img.to(device)

            moved_imgs, affine_mat = reg_net(moving_img, fixed_img)

            sim_loss = sim_criterion(moved_imgs, fixed_img)
            total_loss = args.sim_weight * sim_loss

            run_loss[epoch, 0] += total_loss.item()
            run_loss[epoch, 1] += args.sim_weight * sim_loss.item()

            total_loss.backward()

            optimizer.step()
            optimizer.zero_grad()

        scheduler.step() if args.apply_lr_scheduler else None  # epoch wise lr scheduler

        time_t = time.time() - t0

        if epoch % args.val_interval == 0:
            reg_net.eval()

            for val_idx, (moving_img, fixed_img) in enumerate(val_loader):
                if val_idx > 1: break  # 只查看2个batch

                moving_img, fixed_img = moving_img.to(device), fixed_img.to(device)

                t0 = time.time()

                with torch.no_grad():
                    moved_img, affine_mat = reg_net(moving_img, fixed_img)

                time_i = time.time() - t0

                if args.is_visdom:
                    trans = ImgTransform(scale_type='max-min')
                    moving_img = (trans(moving_img) * 255).type(dtype=torch.uint8)
                    fixed_img = (trans(fixed_img) * 255).type(dtype=torch.uint8)
                    moved_img = (trans(moved_img) * 255).type(dtype=torch.uint8)

                    vis.images(moving_img, nrow=1, win=f'mov_{val_idx}',
                               opts={'title': f'Moving Images {val_idx}',
                                     'width': 100, 'height': 500})
                    vis.images(moved_img, nrow=1, win=f'moved_{val_idx}',
                               opts={'title': f'Moved Images {val_idx}',
                                     'width': 100, 'height': 500})
                    vis.images(fixed_img, nrow=1, win=f'fix_{val_idx}',
                               opts={'title': f'fixed Images {val_idx}',
                                     'width': 100, 'height': 500})

            latest_lr = optimizer.state_dict()['param_groups'][0]['lr']

            np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
            logger.info(
                f"epoch {epoch}, step {steps}, time train {round(time_t, 3)}, time infer {round(time_i, 3)}, "
                f"total loss {run_loss[epoch, 0] :.3f}, sim loss {run_loss[epoch, 1] :.3f}")

            if args.is_visdom:
                # loss line
                vis.line(Y=[run_loss[epoch]], X=[epoch], win='loss', update='append', opts=loss_opts)
                # lr decay line
                vis.line(Y=[latest_lr], X=[epoch], win='lr', update='append', opts=lr_opts)

            if (epoch + 1) % args.save_interval == 0:
                np.save(f"{args.output}run_loss.npy", run_loss)

                state_dict = {
                    "state_dict": reg_net.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict() if args.apply_lr_scheduler else None,
                    "best_acc": best_acc,
                    "epoch": epoch,
                    "steps": steps
                }

                torch.save(state_dict, args.output + f"reg_net_label{args.choose_label}_epoch{epoch}.pth")
                logger.info(f"saved the model at epoch {epoch}")

            reg_net.train()


if __name__ == '__main__':
    main()
