import os
import pathlib
import re
import shutil
import time
import glob
import logging
import argparse

import ants
import torch
import numpy as np
import nibabel as nib

from utils import get_logger
from utils.metrics import dice_coeff, Get_Jac, jacobian_determinant


parser = argparse.ArgumentParser()

# dataset args
parser.add_argument("-dataset", dest="dataset", choices=["tcia", "lpba"], default='tcia', required=False)
parser.add_argument("-img_folder", dest="img_folder", help="training CTs dataset folder",
                    default=r'./dataset/LPBA40/train')
parser.add_argument("-label_folder", dest="label_folder", help="training labels dataset folder",
                    default=r"./dataset/LPBA40/label")
parser.add_argument("-mov_numbers", dest="mov_numbers",
                    help="list of integers indicating which scans to use, i.e. \"1 2 3\" ",
                    default="2 12 17 29",
                    type=lambda s: [n for n in s.split()])
parser.add_argument("-fix_numbers", dest="fix_numbers", type=str,
                    help="list of integers indicating which scans to use, i.e. \"1 2 3\" ",
                    default=1)
parser.add_argument("-img_name", dest="img_name",
                    help="prototype scan filename i.e. pancreas_ct?.nii.gz",
                    default='S?.delineation.skullstripped.nii.gz')  # pancreas_ct?.nii.gz
parser.add_argument("-label_name", dest="label_name", help="prototype segmentation name i.e. label_ct?.nii.gz",
                    default="S?.delineation.structure.label.nii.gz")
parser.add_argument("-trans_typ", dest="trans_typ", help="filename (without extension) for output",
                    default="SyN")
parser.add_argument("-output", dest="output", help="filename (without extension) for output",
                    default="./Result/ANTs_SyN")

args = parser.parse_args()

if not os.path.exists(args.output):
    # os.makedirs(out_dir, exist_ok=True)
    pathlib.Path(args.output).mkdir(parents=True, exist_ok=True)

def get_img_lst(img_path):
    if isinstance(img_path, str):
        # 如果是字符串,那不是图片路径就是文件夹路径
        if img_path.endswith((".nii.gz")):
            img_lst = [img_path]
        else:
            img_lst = glob.glob(os.path.join(img_path, "*.nii.gz"))
    elif isinstance(img_path, list):
        # 如果是列表,我们希望是图片路径列表
        img_lst = img_path
    else:
        raise ValueError(f'unknown filetype for {img_path}')
    return img_lst

def save_image(img, ref_img, name, result_dir):
    # 将配准后图像的direction/origin/spacing和原图保持一致
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    img.set_direction(ref_img.direction)
    img.set_origin(ref_img.origin)
    img.set_spacing(ref_img.spacing)
    ants.image_write(img, os.path.join(result_dir, name))
    print(f"warped img saved to {result_dir}")


def ants_reg(mov_img_p,
             mov_label_p,
             fix_img_p,
             fix_label_p,
             result_dir=args.output,
             trans_typ=args.trans_typ,
             dataset=args.dataset):
    """
    type_of_transform参数的取值可以为:
        Rigid:刚体
        Affine:仿射配准,即刚体+缩放
        ElasticSyN:仿射配准+可变形配准,以MI为优化准则,以elastic为正则项
        SyN:仿射配准+可变形配准,以MI为优化准则
        SyNCC:仿射配准+可变形配准,以CC为优化准则
    """

    logger = get_logger(output=result_dir, name=f'ants_{trans_typ}')

    # ants图片的读取
    f_img = ants.image_read(fix_img_p)
    f_label = ants.image_read(fix_label_p)

    # 我们希望img和label的图片文件名能一一对应
    mov_img_lst = sorted(get_img_lst(mov_img_p))
    mov_label_lst = sorted(get_img_lst(mov_label_p))

    # 指标保存
    num_labels = 55 if dataset.lower() == 'lpba' else 9
    dice_all_val = np.zeros((len(mov_img_lst), num_labels - 1))
    Jac_std1, Jac_neg1 = [], []
    Jac_std2, Jac_neg2 = [], []

    for idx, (mov_img, mov_label) in enumerate(zip(mov_img_lst, mov_label_lst)):
        img_name = re.findall(r'\d+', os.path.split(mov_img)[-1])[0]
        fix_name = re.findall(r'\d+', os.path.split(fix_img_p)[-1])[0]
        save_name = f"{img_name}_{fix_name}"

        m_img = ants.image_read(mov_img)
        m_label = ants.image_read(mov_label)

        # 计算初始指标
        dice_one_val = dice_coeff(f_label.numpy(single_components=False), m_label.numpy(single_components=False),
                                  logger=logger)
        np.set_printoptions(precision=3, suppress=True)
        logger.info(f"{save_name} initial DSC: dice list {dice_one_val}, DSC {dice_one_val.mean()}")

        '''
        ants.registration()函数的返回值是一个字典:
            warpedmovout: 配准到fixed图像后的moving图像矩阵
            warpedfixout: 配准到moving图像后的fixed图像矩阵
            fwdtransforms: 从moving到fixed的形变场的临时文件路径, [0]的.nii.gz是变形场, [1]的.mat文件是仿射位移场
            invtransforms: 从fixed到moving的形变场的临时文件路径, 位置与上面相反
        '''
        t0 = time.time()
        # 图像配准
        mytx = ants.registration(fixed=f_img, moving=m_img, type_of_transform=trans_typ)
        time_o = time.time() - t0

        flow_field = nib.load(mytx['fwdtransforms'][0]).get_fdata().transpose(3, 0, 1, 2, 4)  # shape(batch, H, W, D, channel)

        logger.info(f"flow_filed shape: {flow_field.shape}")

        warped_img = mytx["warpedmovout"]  # mytx["warpedmovout"] 为配准后的 moving img
        # 对moving图像对应的label图进行配准,interpolator也可以选择"nearestNeighbor"等
        warped_label = ants.apply_transforms(fixed=f_label,
                                             moving=m_label,
                                             transformlist=mytx['fwdtransforms'],
                                             interpolator="nearestNeighbor")  # 注意标签需要最邻近插值

        save_image(warped_img, f_img, f"{save_name}_warped_img.nii.gz", result_dir=result_dir)
        save_image(warped_label, f_img, f"{save_name}_warped_label.nii.gz", result_dir=result_dir)
        shutil.copy(mytx['fwdtransforms'][0], os.path.join(result_dir, f"{save_name}_flow.nii.gz"))

        # 计算指标
        dice_one_val = dice_coeff(warped_label.numpy(single_components=False), m_label.numpy(single_components=False),
                                  logger=logger)
        np.set_printoptions(precision=3, suppress=True)
        logger.info(f"{save_name} : dice list {dice_one_val}, DSC {dice_one_val.mean()}")

        Jac = torch.from_numpy(jacobian_determinant(flow_field[0]))  # no needs batch dim
        Jac_std1.append(Jac.std())
        Jac_neg1.append(100 * ((Jac <= 0.).sum() / Jac.numel()))

        Jac = Get_Jac(flow_field)
        Jac_std2.append(Jac.std())
        Jac_neg2.append(100 * ((Jac <= 0.).sum() / Jac.numel()))

        logger.info(
            f"time infer {round(time_o, 3)}, "
            f"jacobian_determinant: stdJac {np.mean(Jac_std1) :.3f}, Jac<=0 {np.mean(Jac_neg1) :.3f}%, "
            f"Get_Jac: stdJac {np.mean(Jac_std2) :.3f}, Jac<=0 {np.mean(Jac_neg2) :.3f}%")

        # 将antsimage转化为numpy数组
        warped_img_arr = warped_img.numpy(single_components=False)
        # 从numpy数组得到antsimage
        img = ants.from_numpy(warped_img_arr, origin=None, spacing=None, direction=None, has_components=False, is_rgb=False)

        # 生成图像的雅克比行列式
        jac = ants.create_jacobian_determinant_image(domain_image=f_img, tx=mytx["fwdtransforms"][0], do_log=False, geom=False)
        ants.image_write(jac, os.path.join(result_dir, f"{save_name}_ants_jac.nii.gz"))

        # 生成带网格的moving图像,可以用于可视化变形场,实测效果不好（为什么要两次 create_grid？）
        m_grid = ants.create_warped_grid(m_img)
        m_grid = ants.create_warped_grid(m_grid, grid_directions=(False, True), transform=mytx['fwdtransforms'],
                                         fixed_reference_image=f_img)
        ants.image_write(m_grid, os.path.join(result_dir, f"{save_name}_m_grid.nii.gz"))

        '''
        以下为其他不常用的函数:

        ANTsTransform.apply_to_image(image, reference=None, interpolation='linear')
        ants.read_transform(filename, dimension=2, precision='float')
        # transform的格式是".mat"
        ants.write_transform(transform, filename)
        # field是ANTsImage类型
        ants.transform_from_displacement_field(field)
        '''

    print("End")


if __name__ == "__main__":

    # 固定图像为一个文件路径，不是列表
    fix_img_p = os.path.join(args.img_folder, args.img_name.replace('?', args.fix_numbers))
    fix_label_p = os.path.join(args.label_folder, args.label_name.replace('?', args.fix_numbers))

    # 浮动图像路径列表
    mov_img_p = [os.path.join(args.img_folder, args.img_name.replace('?', i)) for i in args.mov_numbers]
    mov_label_p = [os.path.join(args.label_folder, args.label_name.replace('?', i)) for i in args.mov_numbers]

    ants_reg(mov_img_p, mov_label_p, fix_img_p, fix_label_p)
