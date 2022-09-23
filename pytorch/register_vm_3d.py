from __future__ import print_function
import numpy as np
import torch
import torch.nn.functional as F
import time
import os
import pathlib
import SimpleITK as sitk
import nibabel as nib

import argparse

cuda_idx = 0

from utils import get_logger, countParam
from utils.metrics import dice_coeff
from datasets import lpba
from models import VxmDense, SpatialTransformer


def main():
    """
    python inference_seg_lpba.py -input preprocess/datasets/process_cts/pancreas_ct1.nii.gz -output mylabel_ct1.nii.gz -groundtruth preprocess/datasets/process_labels/label_ct1.nii.gz
    """
    # read/parse user command line input
    parser = argparse.ArgumentParser()

    parser.add_argument("-dataset", dest="dataset", choices=["tcia", "bcv", "lpba"],
                        help="either tcia or visceral", default='lpba')
    parser.add_argument("-model", dest="model", help="filename of pytorch pth model",
                        default='output/lpba/xxx.pth', )
    parser.add_argument("-with_BN", help="OBELISK Reg_Net with BN or not", action="store_true")

    parser.add_argument("-input", dest="input", help="images folder",
                        default=r"D:\code_sources\from_github\Medical Images Seg & Reg\MICCAI2020\vm_troch\dataset\LPBA40\train", )
    parser.add_argument("-groundtruth", dest="groundtruth", help="labels folder",
                        default=r'D:\code_sources\from_github\Medical Images Seg & Reg\MICCAI2020\vm_troch\dataset\LPBA40\label')
    parser.add_argument("-img_name", dest="img_name",
                        help="prototype scan filename i.e. pancreas_ct?.nii.gz",  # img?_bcv_CT.nii.gz
                        default='S?.delineation.skullstripped.nii.gz')
    parser.add_argument("-label_name", dest="label_name", help="prototype segmentation name i.e. label_ct?.nii.gz",
                        default="S?.delineation.structure.label.nii.gz")
    parser.add_argument("-fix_number", dest="fix_number", help="A number of fixed image",
                        type=lambda s: [int(n) for n in s.split()],
                        default="1")
    parser.add_argument("-mov_numbers", dest="mov_numbers", help="list of numbers of moving images",
                        type=lambda s: [int(n) for n in s.split()],
                        default="2 3 4 5 6 7 8 9")

    parser.add_argument("-output", dest="output", help="nii.gz label output prediction",
                        default="output/lpba_test/")

    args = parser.parse_args()
    d_options = vars(args)
    img_folder = d_options['input']
    label_folder = d_options['groundtruth']
    img_name = d_options['img_name']
    label_name = d_options['label_name']
    output = d_options['output']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = get_logger(output=output, name='test')

    if not os.path.exists(d_options['output']):
        # os.makedirs(out_dir, exist_ok=True)
        pathlib.Path(output).mkdir(parents=True, exist_ok=True)

    # load atlas
    fixed_loader = lpba(
        logger=None,
        img_folder=img_folder,
        img_name=img_name,
        label_folder=label_folder,
        label_name=label_name,
        scannumbers=args.fix_number,
        for_test=True)

    atlas_loader = iter(fixed_loader)
    fixed_img, fixed_label = next(atlas_loader)

    def unet_reg(mov_img, fix_img, mov_seg):
        # load and set up model
        reg_net = VxmDense.load(args.model, device)
        reg_net.to(device)
        reg_net.eval()

        # predict
        with torch.no_grad():
            moved_img, moved_seg, flow_field = reg_net(mov_img, fix_img, mov_seg=mov_seg)

        return moved_img, moved_seg, flow_field

    total_time = []

    def inference(moving_img, moving_label,
                  fixed_img=fixed_img,
                  fixed_label=fixed_label,
                  save_name=''):
        moving_label = moving_label.unsqueeze(1).float()  # [B, C, D, W, H]
        if torch.cuda.is_available() == 1:
            logger.info('using GPU acceleration')
            moving_img = moving_img.cuda()
            moving_label = moving_label.cuda()
            fixed_img, fixed_label = fixed_img.cuda(), fixed_label.cuda()

        t0 = time.time()

        # warped image and label by flow
        moved_img, moved_seg, flow_field = unet_reg(moving_img, fixed_img, moving_label)

        t1 = time.time()
        total_time.append(t1 - t0)
        # if d_options['dataset'] == 'visceral':
        #     predict = F.interpolate(predict, size=[D_in0, H_in0, W_in0], mode='trilinear', align_corners=False)

        save_path = os.path.join(d_options['output'], 'pred?_lpba.nii.gz')

        sitk.WriteImage(sitk.GetImageFromArray(moved_img.squeeze().numpy()),
                        save_path.replace("?", f"{save_name}_warped"))
        sitk.WriteImage(sitk.GetImageFromArray(flow_field.permute(0, 2, 3, 4, 1).squeeze().numpy()),
                        save_path.replace("?", f"{save_name}_flow"))
        sitk.WriteImage(sitk.GetImageFromArray(moved_seg.short().squeeze().numpy()),
                        save_path.replace("?", f"{save_name}_label"))
        logger.info(f"warped scan number {save_name} save to {d_options['output']}")

        dice = dice_coeff(moved_seg.long().cpu(), fixed_label.cpu())
        np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
        logger.info('Dice validation:', dice, 'Avg.', '%0.3f' % (dice.mean()),
                    'Std.', dice.std(), 'time:', np.mean(total_time))

    if os.path.isfile(d_options['input']):
        moving_img = torch.from_numpy(nib.load(d_options['input']).get_fdata()).unsqueeze(0).unsqueeze(0)
        moving_img = (moving_img - moving_img.mean()) / moving_img.std()  # mean-std scale
        if d_options['groundtruth'] is not None:
            moving_label = torch.from_numpy(nib.load(d_options['groundtruth']).get_data()).unsqueeze(0)
        else:
            moving_label = None
        inference(moving_img, moving_label, save_name='')
    elif os.path.isdir(d_options['input']):
        moving_loader = lpba(
            logger=None,
            img_folder=img_folder,
            img_name=img_name,
            label_folder=label_folder,
            label_name=label_name,
            scannumbers=args.mov_numbers,
            for_test=True)

        for idx, (moving_img, moving_label) in enumerate(moving_loader):
            inference(moving_img, moving_label, save_name=str(args.mov_numbers[idx]))


if __name__ == '__main__':
    main()
