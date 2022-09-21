import os
import numpy as np
import torch
import torch.nn as nn
import nibabel as nib
from medpy import metric


def hd95(gt, pred):
    return metric.hd95(result=pred, reference=gt, voxelspacing=1.5)


def dice_simi_coeff(pred, gt, logger=None):
    """
    same as the function below

    :param pred:
    :param gt:
    :param logger:
    :return:
    """
    organ_labels = {0: "background", 1: "liver", 2: "spleen", 3: "r_kidney", 4: "l_kidney"}
    dsc = []
    for i in np.unique(pred)[1:]:
        pred_i = np.where(pred != i, 0., pred)
        dsc.append(metric.dc(result=pred_i, reference=gt))
        if logger:
            try:
                logger.info(f"{organ_labels[i]}: {dsc[i - 1] :.3f}")
            except:
                pass
    return dsc


def dice_coeff(outputs, labels, logger=None):
    """
    Evaluation function for Dice score of segmentation overlap
    """
    organ_labels = {0: "background", 1: "liver", 2: "spleen", 3: "r_kidney", 4: "l_kidney"}
    label_nums = np.unique(labels)
    # print("labels:", label_nums)
    dice = []
    for label in label_nums[1:]:
        iflat = (outputs == label).reshape(-1).float()
        tflat = (labels == label).reshape(-1).float()
        intersection = (iflat * tflat).sum()
        dsc = (2. * intersection) / (iflat.sum() + tflat.sum())
        if logger:
            try:
                logger.info(f"{organ_labels[label]}: {dsc :.3f}")
            except:
                pass
        dice.append(dsc)
    return np.asarray(dice)
    # return metric.dc(result=outputs, reference=labels)


def Get_Jac(displacement):
    '''
    compute the Jacobian determinant to find out the smoothness of the u.
    refer: https://blog.csdn.net/weixin_41699811/article/details/87691884

    Param: displacement of shape(batch, H, W, D, channel)
    '''
    D_y = (displacement[:, 1:, :-1, :-1, :] - displacement[:, :-1, :-1, :-1, :])
    D_x = (displacement[:, :-1, 1:, :-1, :] - displacement[:, :-1, :-1, :-1, :])
    D_z = (displacement[:, :-1, :-1, 1:, :] - displacement[:, :-1, :-1, :-1, :])

    D1 = (D_x[..., 0] + 1) * ((D_y[..., 1] + 1) * (D_z[..., 2] + 1) - D_y[..., 2] * D_z[..., 1])
    D2 = (D_x[..., 1]) * (D_y[..., 0] * (D_z[..., 2] + 1) - D_y[..., 2] * D_z[..., 0])
    D3 = (D_x[..., 2]) * (D_y[..., 0] * D_z[..., 1] - (D_y[..., 1] + 1) * D_z[..., 0])

    D = D1 - D2 + D3

    return D


# compute jacobian determinant as measure of deformation complexity
def jacobian_determinant_3d(dense_flow):
    B, _, H, W, D = dense_flow.size()

    dense_pix = dense_flow * (torch.Tensor([H - 1, W - 1, D - 1]) / 2).view(1, 3, 1, 1, 1).to(dense_flow.device)
    gradz = nn.Conv3d(3, 3, (3, 1, 1), padding=(1, 0, 0), bias=False, groups=3)
    gradz.weight.data[:, 0, :, 0, 0] = torch.tensor([-0.5, 0, 0.5]).view(1, 3).repeat(3, 1)
    gradz.to(dense_flow.device)
    grady = nn.Conv3d(3, 3, (1, 3, 1), padding=(0, 1, 0), bias=False, groups=3)
    grady.weight.data[:, 0, 0, :, 0] = torch.tensor([-0.5, 0, 0.5]).view(1, 3).repeat(3, 1)
    grady.to(dense_flow.device)
    gradx = nn.Conv3d(3, 3, (1, 1, 3), padding=(0, 0, 1), bias=False, groups=3)
    gradx.weight.data[:, 0, 0, 0, :] = torch.tensor([-0.5, 0, 0.5]).view(1, 3).repeat(3, 1)
    gradx.to(dense_flow.device)
    with torch.no_grad():
        jacobian = torch.cat((gradz(dense_pix), grady(dense_pix), gradx(dense_pix)), 0) + \
                   torch.eye(3, 3).view(3, 3, 1, 1, 1).to(dense_flow.device)
        jacobian = jacobian[:, :, 2:-2, 2:-2, 2:-2]
        jac_det = jacobian[0, 0, :, :, :] * (jacobian[1, 1, :, :, :] * jacobian[2, 2, :, :, :] -
                                             jacobian[1, 2, :, :, :] * jacobian[2, 1, :, :, :]) - \
                  jacobian[1, 0, :, :, :] * (jacobian[0, 1, :, :, :] * jacobian[2, 2, :, :, :] -
                                             jacobian[0, 2, :, :, :] * jacobian[2, 1, :, :, :]) + \
                  jacobian[2, 0, :, :, :] * (jacobian[0, 1, :, :, :] * jacobian[1, 2, :, :, :] -
                                             jacobian[0, 2, :, :, :] * jacobian[1, 1, :, :, :])

    return jac_det


if __name__ == '__main__':
    from datasets import CT2MRDataset
    from torch.utils.data import DataLoader

    ct2mr_dataset = CT2MRDataset(
        CT_folder=r"F:\shb_src\from_github\datasets\MICCAI2021\task1_preprocessed_by_shb\auxiliary\L2R_Task1_CT",
        CT_name=r"?_bcv_CT.nii.gz",
        MR_folder=r"F:\shb_src\from_github\datasets\MICCAI2021\task1_preprocessed_by_shb\auxiliary\L2R_Task1_MR",
        MR_name=r"?_chaos_MR.nii.gz",
        pair_numbers=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32],
        for_inf=True
    )

    my_dataloader = DataLoader(dataset=ct2mr_dataset, batch_size=1, num_workers=2)
    dice_all_val = np.zeros((len(ct2mr_dataset), 5 - 1))
    for idx, (CTimgs, CTsegs, MRimgs, MRsegs, CTaffines, MRaffines) in enumerate(my_dataloader):
        dice_all_val[idx] = dice_coeff(MRsegs, CTsegs)
    all_val_dice_avgs = dice_all_val.mean(axis=0)
    mean_all_dice = all_val_dice_avgs.mean()
    np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
    print(f"dice_all_val: {dice_all_val}, \n "
          f"all_val_dice_avgs: {all_val_dice_avgs}, \n "
          f"mean_all_dice: {mean_all_dice}")
