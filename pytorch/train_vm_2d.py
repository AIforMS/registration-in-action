import os
import time
import argparse
import pathlib
import numpy as np
import visdom

import torch
import torch.nn as nn
from torch import optim

from utils import get_logger, countParam, LinearWarmupCosineAnnealingLR
from datasets import mnist
from utils.losses import gradient_loss, NCCLoss
from utils.metrics import jacobian_determinant_3d
from models import VxmDense, SpatialTransformer


def main():
    # read/parse user command line input
    parser = argparse.ArgumentParser()

    # dataset args
    parser.add_argument("-output", help="filename (without extension) for output",
                        default="output/pair-wise-mmr/")
    parser.add_argument("-val_size", help="validation set size, which is divided from the training set",
                        type=int, default=512)
    parser.add_argument("-choose_label", help="Which number to choose for registration training",
                        type=int, default=5)

    # training args
    parser.add_argument("-batch_size", help="Dataloader batch size", type=int, default=1)
    parser.add_argument("-lr", help="Optimizer learning rate, keep pace with batch_size",
                        type=float, default=1e-4)  # 0.005 for AdamW, 4e-4 for Adam
    parser.add_argument("-apply_lr_scheduler", help="Need lr scheduler or not", action="store_true")
    parser.add_argument("-warmup_epochs", help="epochs for Warmup scheduler", type=int, default=10)
    parser.add_argument("-epochs", help="Train epochs",
                        type=int, default=500)
    parser.add_argument("-resume", help="Path to pretrained model to continute training",
                        default=None)  # "output/LPBA40_noBN/lpba40_best63.pth"
    parser.add_argument("-interval", help="validation and saving interval", type=int, default=5)
    parser.add_argument("-is_visdom", help="Using Visdom to visualize Training process",
                        type=lambda s: False if s == "False" else True, default=False)
    parser.add_argument("-num_workers", help="Dataloader num_workers", type=int, default=2)

    # losses args
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
    if args.weakly_sup:
        logger.info("Weakly supervised training with dice loss")
    logger.info(f"output to {args.output}")

    train_loader = mnist(for_what='train',
                         batch_size=args.batch_size,
                         val_size=args.val_size,
                         choose_label=args.choose_label,
                         num_workers=args.num_workers)

    val_loader = mnist(for_what='val',
                       batch_size=args.batch_size,
                       val_size=args.val_size,
                       choose_label=args.choose_label,
                       num_workers=args.num_workers)

    end_epoch = args.epochs  # 300

    # initialise trainable network parts
    enc_nf = [16, 32, 32, 32]
    dec_nf = [32, 32, 32, 32, 32, 16, 16]
    reg_net = VxmDense(
        inshape=[32, 32],
        nb_unet_features=[enc_nf, dec_nf],
        int_steps=args.int_steps,
        int_downsize=args.int_downsize)
    reg_net.to(device)

    logger.info(f'VM reg_net params: {countParam(reg_net)}')

    if args.resume:
        reg_net = reg_net.load(args.resume, device=device).to(device)
        logger.info(f"Training resume from {args.resume}")

    reg_net.train()

    stn = SpatialTransformer(size=[32, 32])
    stn.to(device)

    # train using Adam with weight decay and exponential LR decay
    optimizer = optim.Adam(reg_net.parameters(), lr=args.lr)
    # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999) if args.apply_lr_scheduler else None
    scheduler = LinearWarmupCosineAnnealingLR(optimizer,
                                              warmup_epochs=args.warmup_epochs,
                                              max_epochs=args.epochs) if args.apply_lr_scheduler else None

    # losses
    if args.sim_loss == "MSE":
        sim_criterion = nn.MSELoss()
        args.alpha = 0.025
    elif args.sim_loss == "NCC":
        sim_criterion = NCCLoss()
        args.alpha = 1.5
    grad_criterion = gradient_loss

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
        best_acc_opt = {'xlabel': 'epochs', 'ylabel': 'best acc', 'title': 'Best Acc Line'}

    for epoch in range(end_epoch):
        t0 = time.time()
        for moving_img, fixed_img in train_loader:
            steps += 1

            moving_img, fixed_img = moving_img.to(device), fixed_img.to(device)

            moved_imgs, flow_field = reg_net(moving_img, fixed_img)

            sim_loss = sim_criterion(moved_imgs, fixed_img)
            grad_loss = grad_criterion(flow_field)
            total_loss = args.sim_weight * sim_loss + args.alpha * grad_loss

            run_loss[epoch, 0] += total_loss.item()
            run_loss[epoch, 1] += args.sim_weight * sim_loss.item()
            run_loss[epoch, 2] += args.alpha * grad_loss.item()

            total_loss.backward()

            optimizer.step()
            optimizer.zero_grad()

        scheduler.step() if scheduler else None  # epoch wise lr scheduler

        time_t = time.time() - t0

        if epoch % args.interval == 0:
            reg_net.eval()
            Jac_std, Jac_neg = [], []
            images = None

            for val_idx, (moving_img, fixed_img) in enumerate(val_loader):
                moving_img, fixed_img = moving_img.to(device), fixed_img.to(device)

                t0 = time.time()

                with torch.no_grad():
                    moved_img, flow_field = reg_net(moving_img,
                                                     fixed_img)

                    time_i = time.time() - t0

                    # complexity of transformation and foldings
                    jacdet = jacobian_determinant_3d(flow_field).cpu()
                    Jac_std.append(jacdet.std())
                    Jac_neg.append(100 * ((jacdet <= 0.).sum() / jacdet.numel()))

            latest_lr = optimizer.state_dict()['param_groups'][0]['lr']

            np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
            logger.info(
                f"epoch {epoch}, step {steps}, time train {round(time_t, 3)}, time infer {round(time_i, 3)}, "
                f"total loss {run_loss[epoch, 0] :.3f}, sim loss {run_loss[epoch, 1] :.3f}, "
                f"grad loss {run_loss[epoch, 2] :.3f}, stdJac {np.mean(Jac_std) :.3f}, Jac<=0 {np.mean(Jac_neg) :.3f}%")

            if args.is_visdom:
                # loss line
                vis.line(Y=[run_loss[epoch]], X=[epoch], win='loss', update='append', opts=loss_opts)
                # lr decay line
                vis.line(Y=[latest_lr], X=[epoch], win='lr', update='append', opts=lr_opts)
                vis.images()

            np.save(f"{args.output}run_loss.npy", run_loss)
            reg_net.save(args.output + f"{args.dataset}_best.pth")
            logger.info(f"saved the best model at epoch {epoch}")

            reg_net.train()


if __name__ == '__main__':
    main()
