#!/usr/bin/env python
import os
import argparse
import pathlib
import numpy as np
import nibabel as nib
import torch
import visdom

from datasets import mnist
from utils.utils import ImgTransform, get_logger
from models import VxmDense

# parse commandline args
parser = argparse.ArgumentParser()

# dataset args
parser.add_argument("-output", help="filename (without extension) for output",
                    default="output/mnist_test/")
parser.add_argument("-choose_label", help="Which number to choose for registration training",
                    type=int, default=7)

# testing args
parser.add_argument("-batch_size", help="Dataloader batch size", type=int, default=4)
parser.add_argument("-model", help="Path to pretrained model to continute training",
                    default="output/mnist/reg_net_299.pth")  # "output/mnist/reg_net_199.pth"
parser.add_argument("-is_visdom", help="Using Visdom to visualize Training process",
                    type=lambda s: False if s == "False" else True, default=True)
parser.add_argument("-num_workers", help="Dataloader num_workers", type=int, default=2)
parser.add_argument("-save_images", help="Will you wanna save images?",
                    type=lambda s: True if s == 'True' else False, default=True)

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not os.path.exists(args.output):
    pathlib.Path(os.path.join(args.output, 'flow_fields')).mkdir(parents=True, exist_ok=True)

if args.is_visdom:
    # 需要提前在终端运行服务: python -m visdom.server
    vis = visdom.Visdom()

logger = get_logger(args.output, name='test')


def vm_reg(mov_img, fix_img):
    # load and set up model
    reg_net = VxmDense.load(args.model, device)
    reg_net.to(device)
    reg_net.eval()

    # predict
    with torch.no_grad():
        moved_img, flow_field = reg_net(mov_img, fix_img)

    return moved_img, flow_field


def main():

    # load images
    test_loader = mnist(for_what='test',
                        choose_label=args.choose_label,
                        batch_size=args.batch_size,
                        num_workers=args.num_workers)

    for val_idx, (moving_img, fixed_img) in enumerate(test_loader):
        if val_idx > 2: break  # 只循环3次

        moving_img, fixed_img = moving_img.to(device), fixed_img.to(device)
        moved_img, flow_field = vm_reg(moving_img, fixed_img)

        # visualization
        if args.is_visdom:
            trans = ImgTransform(scale_type='max-min')
            moving_img = (trans(moving_img) * 255).type(dtype=torch.uint8)
            fixed_img = (trans(fixed_img) * 255).type(dtype=torch.uint8)
            moved_img = (trans(moved_img) * 255).type(dtype=torch.uint8)
            flow_field = (trans(flow_field) * 255).type(dtype=torch.uint8)

            vis.images(moving_img, nrow=1, win=f'mov_{val_idx}',
                       opts={'title': f'Moving Images {val_idx}',
                             'width': 100, 'height': 500})
            vis.images(moved_img, nrow=1, win=f'moved_{val_idx}',
                       opts={'title': f'Moved Images {val_idx}',
                             'width': 100, 'height': 500})
            vis.images(fixed_img, nrow=1, win=f'fix_{val_idx}',
                       opts={'title': f'fixed Images {val_idx}',
                             'width': 100, 'height': 500})

        if args.save_images:
            # flow_field 无法通过 plt 可视化，需要保存成 nii.gz, 使用 ITK-SNAP 打开
            nib.save(nib.Nifti1Image(flow_field.cpu().permute(0, 2, 3, 1).squeeze().numpy(), np.eye(4)),
                     os.path.join(args.output, 'flow_fields', f"flow_field{val_idx}.nii.gz"))

if __name__ == "__main__":
    main()
