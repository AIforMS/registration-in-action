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


def jacobian_determinant(disp):
    """
    jacobian determinant of a displacement field. From VoxelMorph
    NB: to compute the spatial gradients, we use np.gradient.
    Parameters:
        disp: 2D or 3D displacement field of size [*vol_shape, nb_dims],
              where vol_shape is of len nb_dims
    Returns:
        jacobian determinant (scalar)
    """
    if len(disp.shape) not in [3, 4]:
        raise ValueError(f"shape of 2D folw field needs to be [H, W, C]，"
                         f"shape of 3D folw field needs to be [H, W, D, C], but got {disp.shape}")
    # check inputs
    volshape = disp.shape[:-1]
    nb_dims = len(volshape)
    assert len(volshape) in (2, 3), 'flow has to be 2D or 3D'

    # compute grid
    grid_lst = [np.arange(e) for e in volshape]
    grid_lst = np.meshgrid(*grid_lst, indexing='ij')
    grid = np.stack(grid_lst, len(volshape))

    # compute gradients
    J = np.gradient(disp + grid)

    # 3D glow
    if nb_dims == 3:
        dx = J[0]
        dy = J[1]
        dz = J[2]

        # compute jacobian components
        Jdet0 = dx[..., 0] * (dy[..., 1] * dz[..., 2] - dy[..., 2] * dz[..., 1])
        Jdet1 = dx[..., 1] * (dy[..., 0] * dz[..., 2] - dy[..., 2] * dz[..., 0])
        Jdet2 = dx[..., 2] * (dy[..., 0] * dz[..., 1] - dy[..., 1] * dz[..., 0])

        return Jdet0 - Jdet1 + Jdet2

    else:  # must be 2

        dfdx = J[0]
        dfdy = J[1]

        return dfdx[..., 0] * dfdy[..., 1] - dfdy[..., 0] * dfdx[..., 1]
