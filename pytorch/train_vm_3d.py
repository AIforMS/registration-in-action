import os
import time
import argparse
import pathlib
import numpy as np
import visdom

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from utils import get_logger, countParam, LinearWarmupCosineAnnealingLR, augment_affine, setup_seed
from utils.losses import gradient_loss, NCCLoss, MIND_loss, DiceCELoss
from utils.metrics import dice_coeff, Get_Jac
from datasets import lpba
from models import VxmDense, SpatialTransformer

setup_seed()


def main():
    # read/parse user command line input
    parser = argparse.ArgumentParser()

    # dataset args
    parser.add_argument("-dataset", choices=["lpba"], default='lpba')

    parser.add_argument("-img_folder", help="training images folder",
                        default=r'F:\shb_src\from_github\OBELISK\preprocess\datasets\LPBA40\train')
    parser.add_argument("-img_name", help="prototype scan filename",
                        default='S?.delineation.skullstripped.nii.gz')
    parser.add_argument("-label_folder", help="training labels dataset folder",
                        default=r'F:\shb_src\from_github\OBELISK\preprocess\datasets\LPBA40\label')
    parser.add_argument("-label_name", help="prototype label filename",
                        default='S?.delineation.structure.label.nii.gz')

    parser.add_argument("-train_scannumbers",
                        help="list of integers indicating which scans to use, i.e. \"1 2 3\" ",
                        default="4 5 6 7 8 9 10 11 12 13 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 34 35 36 37 38 39 40",
                        type=lambda s: [int(n) for n in s.split()])
    parser.add_argument("-val_scannumbers",
                        help="list of integers indicating which scans to use, i.e. \"1 2 3\" ",
                        default="1 2 3",
                        type=lambda s: [int(n) for n in s.split()])

    parser.add_argument("-output", help="filename (without extension) for output",
                        default="output/lpba/")

    # training args
    parser.add_argument("-int_steps", help="Number of flow integration steps. "
                                           "The warp is non-diffeomorphic when this value is 0.",
                        type=int, default=7)
    parser.add_argument("-int_downsize", help="Integer specifying the flow downsample factor for vector integration. "
                                              "The flow field is not downsampled when this value is 1.",
                        type=int, default=2)
    parser.add_argument("-corner", help="corner", type=bool, default=False)

    parser.add_argument("-batch_size", help="Dataloader batch size", type=int, default=1)
    parser.add_argument("-step_interval", help="implicit the batchsize", type=int, default=4)
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
    parser.add_argument("-weakly_sup", help="if apply weakly supervised, use reg dice loss, else not",
                        action="store_true")
    parser.add_argument("-sim_loss", type=str, help="similarity criterion", choices=['MIND', 'MSE', 'NCC'],
                        default='MSE')
    parser.add_argument("-alpha", type=float, help="weight for regularization loss",
                        default=0.025)  # ncc: 1.5, mse: 0.025, MIND-SSC: 4.0, VM: 0.01
    parser.add_argument("-dice_weight", help="Dice loss weight",
                        type=float, default=0.1)
    parser.add_argument("-sim_weight", help="CE loss weight",
                        type=float, default=1.0)

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(args.output):
        pathlib.Path(args.output).mkdir(parents=True, exist_ok=True)

    logger = get_logger(args.output)
    if args.weakly_sup:
        logger.info("Weakly supervised training with dice loss")
    logger.info(f"output to {args.output}")

    # load train images and segmentations
    logger.info(f'train scan numbers: {args.train_scannumbers}')

    train_loader, val_loader, num_labels = lpba(logger=logger,
                                                img_folder=args.img_folder,
                                                img_name=args.img_name,
                                                label_folder=args.label_folder,
                                                label_name=args.label_name,
                                                train_scannumbers=args.train_scannumbers,
                                                val_scannumbers=args.val_scannumbers,
                                                batch_size=args.batch_size,
                                                num_workers=args.num_workers)

    end_epoch = args.epochs  # 300

    logger.info(f"num of labels: {num_labels}")

    img_shape = [160, 192, 160]

    # initialise trainable network parts
    enc_nf = [16, 32, 32, 32]
    dec_nf = [32, 32, 32, 32, 32, 16, 16]
    reg_net = VxmDense(
        inshape=img_shape,
        nb_unet_features=[enc_nf, dec_nf],
        int_steps=args.int_steps,
        int_downsize=args.int_downsize)
    reg_net.to(device)
    logger.info(f'VM reg_net params: {countParam(reg_net)}')

    if args.resume:
        reg_net = reg_net.load(args.resume, device=device).to(device)
        logger.info(f"Training resume from {args.resume}")

    reg_net.train()

    stn_val = SpatialTransformer(size=img_shape)
    stn_val.to(device)

    # train using Adam with weight decay and exponential LR decay
    optimizer = optim.Adam(reg_net.parameters(), lr=args.lr)
    # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999) if args.apply_lr_scheduler else None
    scheduler = LinearWarmupCosineAnnealingLR(optimizer,
                                              warmup_epochs=args.warmup_epochs,
                                              max_epochs=args.epochs) if args.apply_lr_scheduler else None

    # losses
    if args.sim_loss == "MIND":
        sim_criterion = MIND_loss
        args.alpha = 4.0
    elif args.sim_loss == "MSE":
        sim_criterion = nn.MSELoss()
        args.alpha = 0.025
    elif args.sim_loss == "NCC":
        sim_criterion = NCCLoss()
        args.alpha = 1.5
    grad_criterion = gradient_loss
    dce_criterion = DiceCELoss(to_onehot_y=True, softmax=True)  # dice_loss
    ce_criterion = nn.CrossEntropyLoss()

    steps, best_acc = 0, 0
    run_loss = np.zeros([end_epoch, 4])
    dice_all_val = np.zeros((len(args.val_scannumbers), num_labels - 1))

    if args.is_visdom:
        vis = visdom.Visdom()  # using visdom
        logger.info("visdom starting, need to open the server: python -m visdom.server")
        loss_opts = {'xlabel': 'epochs',
                     'ylabel': 'loss',
                     'title': 'Loss Line',
                     'legend': ['total loss', 'sim loss', 'dice loss', 'grad loss']}
        acc_opts = {'xlabel': 'epochs',
                    'ylabel': 'acc',
                    'title': 'Acc Line',
                    'legend': ['1 spleen', '2 pancreas', '3 kidney', '4 gallbladder', '5 esophagus', '6 liver',
                               '7 stomach', '8 duodenum'] if args.dataset == 'tcia'
                    else ['1 liver', '2 spleen', '3 right kidney', '4 left kidney']}
        lr_opts = {'xlabel': 'epochs', 'ylabel': 'lr', 'title': 'LR Line'}
        best_acc_opt = {'xlabel': 'epochs', 'ylabel': 'best acc', 'title': 'Best Acc Line'}

    # run for 1000 iterations / 250 epochs
    for epoch in range(end_epoch):

        t0 = time.time()

        # select random training pair (mini-batch=4 averaging at the end)
        for moving_imgs, moving_labels, fixed_imgs, fixed_labels in train_loader:
            steps += 1

            if np.random.choice([0, 1]):
                # 50% to apply data augment
                with torch.no_grad():
                    moving_imgs, moving_labels = augment_affine(moving_imgs.to(device),
                                                                moving_labels.to(device),
                                                                strength=0.0375)
                    fixed_imgs, fixed_labels = augment_affine(fixed_imgs.to(device),
                                                              fixed_labels.to(device),
                                                              strength=0.0375)
                    torch.cuda.empty_cache()
            else:
                moving_imgs, moving_labels = moving_imgs.to(device), moving_labels.to(device)
                fixed_imgs, fixed_labels = fixed_imgs.to(device), fixed_labels.to(device)

            # run forward path with previous weights
            moved_imgs, flow_field = reg_net(moving_imgs, fixed_imgs)

            # Pytorch grid_sample用最近邻插值梯度会是0。如果用线性插值的话，不能直接插原label，要先one-hot。
            moving_labels_one_hot = F.one_hot(
                moving_labels.squeeze(1).long(), num_classes=num_labels).permute(0, 4, 1, 2, 3).float()  # NxNum_LabelsxHxWxD

            moved_labels = stn_val(moving_labels_one_hot, flow_field)  # 采用线性插值对seg进行warped

            sim_loss = sim_criterion(moved_imgs, fixed_imgs)
            grad_loss = grad_criterion(flow_field)
            dce_loss = ce_criterion(moved_labels, fixed_labels)
            total_loss = args.sim_weight * sim_loss \
                         + args.alpha * grad_loss \
                         + args.dice_weight * dce_loss

            run_loss[epoch, 0] += total_loss.item()
            run_loss[epoch, 1] += args.sim_weight * sim_loss.item()
            run_loss[epoch, 2] += args.alpha * grad_loss.item()
            run_loss[epoch, 3] += args.dice_weight * dce_loss.item()
            total_loss.backward()

            # implicit mini-batch of 4 (and LR-decay)
            if args.batch_size == 1 and steps % args.step_interval == 0:
                optimizer.step()
                optimizer.zero_grad()

        scheduler.step() if scheduler else None  # epoch wise lr scheduler

        time_t = time.time() - t0

        # verbose ON: report some numbers and run inference on (potentially unseen test images)
        if epoch % args.interval == 0:
            reg_net.eval()
            Jac_std, Jac_neg = [], []

            for val_idx, (moving_img, moving_label, fixed_img, fixed_label) in enumerate(val_loader):
                moving_img, moving_label, fixed_img = moving_img.to(device), moving_label.to(device), fixed_img.to(device)
                t0 = time.time()
                with torch.no_grad():
                    moved_img, moved_label, flow_field = reg_net(moving_img,
                                                                 fixed_img,
                                                                 mov_seg=moving_label.unsqueeze(0).float())
                    time_i = time.time() - t0

                    dice_all_val[val_idx] = dice_coeff(fixed_label.cpu(), moved_label.long().cpu())

                    # complexity of transformation and foldings
                    jacdet = Get_Jac(flow_field.permute(0, 2, 3, 4, 1)).cpu()
                    Jac_std.append(jacdet.std())
                    Jac_neg.append(100 * ((jacdet <= 0.).sum() / jacdet.numel()))

            # logger some feedback information
            all_val_dice_avgs = dice_all_val.mean(axis=0)
            mean_all_dice = all_val_dice_avgs.mean()
            latest_lr = optimizer.state_dict()['param_groups'][0]['lr']

            is_best = mean_all_dice > best_acc
            best_acc = max(mean_all_dice, best_acc)

            np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
            logger.info(
                f"epoch {epoch}, step {steps}, time train {round(time_t, 3)}, time infer {round(time_i, 3)}, "
                f"total loss {run_loss[epoch, 0] :.3f}, sim loss {run_loss[epoch, 1] :.3f}, "
                f"grad loss {run_loss[epoch, 2] :.3f}, ce loss {run_loss[epoch, 3] :.3f}, "
                f"stdJac {np.mean(Jac_std) :.3f}, Jac<=0 {np.mean(Jac_neg) :.3f}%, "
                f"all_val_dice_avgs {all_val_dice_avgs}, dice avgs {mean_all_dice :.3f}, best_acc {best_acc :.3f}, lr {latest_lr :.8f}")

            if args.is_visdom:
                # loss line
                vis.line(Y=[run_loss[epoch]], X=[epoch], win='loss', update='append', opts=loss_opts)
                # acc line
                # vis.line(Y=[all_val_dice_avgs], X=[epoch], win='acc+', update='append', opts=acc_opts)
                vis.line(Y=[mean_all_dice], X=[epoch], win='best_acc', update='append', opts=best_acc_opt)
                # lr decay line
                vis.line(Y=[latest_lr], X=[epoch], win='lr', update='append', opts=lr_opts)

            if is_best:
                np.save(f"{args.output}run_loss.npy", run_loss)
                reg_net.save(args.output + f"{args.dataset}_best_{round(best_acc, 3) if best_acc > 0.5 else ''}.pth")
                logger.info(f"saved the best model at epoch {epoch}, with best acc {best_acc :.3f}")
                if args.is_visdom:
                    vis.line(Y=[best_acc], X=[epoch], win='best_acc+', update='append', opts=best_acc_opt)

            reg_net.train()


if __name__ == '__main__':
    main()
