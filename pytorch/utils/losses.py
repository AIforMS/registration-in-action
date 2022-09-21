import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import DiceCELoss

import warnings

warnings.filterwarnings("ignore")


class OHEMLoss(torch.nn.NLLLoss):
    """ Online Hard Example Mining Loss.
    Needs input from nn.LogSoftmax() """

    def __init__(self, ratio, weights):
        super(OHEMLoss, self).__init__(None, True)
        self.ratio = ratio
        self.weights = weights

    def forward(self, x, y):
        if len(x.size()) == 5:
            x = x.permute(0, 2, 3, 4, 1).contiguous().view(-1, x.size(1))
        if len(x.size()) == 4:
            x = x.permute(0, 2, 3, 1).contiguous().view(-1, x.size(1))
        if len(x.size()) == 3:
            x = x.permute(0, 2, 1).contiguous().view(-1, x.size(1))
        y = y.reshape(-1)
        num_inst = x.size(0)
        num_hns = int(self.ratio * num_inst)
        x_ = x.clone()
        inst_losses = F.cross_entropy(x_, y, reduce=False)
        _, idxs = inst_losses.topk(num_hns)
        x_hn = x.index_select(0, idxs)
        y_hn = y.index_select(0, idxs)
        return torch.nn.functional.nll_loss(x_hn, y_hn, weight=self.weights)


def multi_class_dice_loss(soft_pred, target, num_labels, weights=None):
    loss = 0
    target = target.float()
    smooth = 1e-6
    for i in range(num_labels):
        score = soft_pred[:, i]
        target_ = target == i
        intersect = torch.sum(score * target_)
        y_sum = torch.sum(target_ * target_)
        z_sum = torch.sum(score * score)
        loss += ((2 * intersect + smooth) / (z_sum + y_sum + smooth))
        if weights is not None:
            loss *= weights[i]
    loss = 1 - (loss / num_labels)
    return loss


class NCCLoss(nn.Module):
    """
    local (over window) normalized cross correlation
    """

    def __init__(self, win=3, eps=1e-8):
        super(NCCLoss, self).__init__()
        self.win = win
        self.eps = eps
        self.w_temp = win

    def forward(self, I, J):
        ndims = 3
        win_size = self.w_temp

        # set window size
        if self.win is None:
            self.win = [5] * ndims
        else:
            self.win = [self.w_temp] * ndims

        weight_win_size = self.w_temp
        weight = torch.ones((1, 1, weight_win_size, weight_win_size, weight_win_size), device=I.device,
                            requires_grad=False)
        conv_fn = F.conv3d

        # compute CC squares
        I2 = I * I
        J2 = J * J
        IJ = I * J

        # compute filters
        # compute local sums via convolution
        I_sum = conv_fn(I, weight, padding=int(win_size / 2))
        J_sum = conv_fn(J, weight, padding=int(win_size / 2))
        I2_sum = conv_fn(I2, weight, padding=int(win_size / 2))
        J2_sum = conv_fn(J2, weight, padding=int(win_size / 2))
        IJ_sum = conv_fn(IJ, weight, padding=int(win_size / 2))

        # compute cross correlation
        win_size = np.prod(self.win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + self.eps)

        # return negative cc.
        return -1.0 * torch.mean(cc)


def gradient_loss(s, penalty='l2'):
    """
    displacement regularization loss
    """
    dy = torch.abs(s[:, :, 1:, :, :] - s[:, :, :-1, :, :])
    dx = torch.abs(s[:, :, :, 1:, :] - s[:, :, :, :-1, :])
    dz = torch.abs(s[:, :, :, :, 1:] - s[:, :, :, :, :-1])

    if (penalty == 'l2'):
        dy = dy * dy
        dx = dx * dx
        dz = dz * dz

    d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
    return d / 3.0


def pdist_squared(x):
    xx = (x ** 2).sum(dim=1).unsqueeze(2)
    yy = xx.permute(0, 2, 1)
    dist = xx + yy - 2.0 * torch.bmm(x.permute(0, 2, 1), x)
    dist[dist != dist] = 0
    dist = torch.clamp(dist, 0.0, 255.0)
    return dist


def MINDSSC(img, radius=3, dilation=3, device=torch.device('cuda')):
    """
    *Preliminary* pytorch implementation.
    MIND-SSC Losses for VoxelMorph
    """
    # see http://mpheinrich.de/pub/miccai2013_943_mheinrich.pdf for details on the MIND-SSC descriptor

    # kernel size
    kernel_size = radius * 2 + 1

    # define start and end locations for self-similarity pattern
    six_neighbourhood = torch.tensor([[0, 1, 1],
                                      [1, 1, 0],
                                      [1, 0, 1],
                                      [1, 1, 2],
                                      [2, 1, 1],
                                      [1, 2, 1]]).long()

    # squared distances
    dist = pdist_squared(six_neighbourhood.t().unsqueeze(0)).squeeze(0)

    # define comparison mask
    x, y = torch.meshgrid(torch.arange(6), torch.arange(6))
    mask = ((x > y).view(-1) & (dist == 2).view(-1))

    # build kernel
    idx_shift1 = six_neighbourhood.unsqueeze(1).repeat(1, 6, 1).view(-1, 3)[mask, :]
    idx_shift2 = six_neighbourhood.unsqueeze(0).repeat(6, 1, 1).view(-1, 3)[mask, :]
    mshift1 = torch.zeros(12, 1, 3, 3, 3).to(device)
    mshift1.view(-1)[torch.arange(12) * 27 + idx_shift1[:, 0] * 9 + idx_shift1[:, 1] * 3 + idx_shift1[:, 2]] = 1
    mshift2 = torch.zeros(12, 1, 3, 3, 3).to(device)
    mshift2.view(-1)[torch.arange(12) * 27 + idx_shift2[:, 0] * 9 + idx_shift2[:, 1] * 3 + idx_shift2[:, 2]] = 1
    rpad1 = nn.ReplicationPad3d(dilation)
    rpad2 = nn.ReplicationPad3d(radius)

    # compute patch-ssd
    ssd = F.avg_pool3d(rpad2(
        (F.conv3d(rpad1(img), mshift1, dilation=dilation) - F.conv3d(rpad1(img), mshift2, dilation=dilation)) ** 2),
        kernel_size, stride=1)

    # MIND equation
    mind = ssd - torch.min(ssd, 1, keepdim=True)[0]
    mind_var = torch.mean(mind, 1, keepdim=True)
    mind_var = mind_var.cpu().data
    mind_var = torch.clamp(mind_var, mind_var.mean() * 0.001, mind_var.mean() * 1000)

    mind_var = mind_var.to(device)  # .to(device)
    mind /= mind_var
    mind = torch.exp(-mind)

    # permute to have same ordering as C++ code
    mind = mind[:, torch.tensor([6, 8, 1, 11, 2, 10, 0, 7, 9, 4, 5, 3]).long(), :, :, :]

    return mind  # Tensor: (N, 12, 192, 160, 192)


def MIND_loss(x, y):
    """
    The loss is small, even the voxel intensity distribution of fake image is so difference, loss.item < 0.14
    """
    return torch.mean((MINDSSC(x) - MINDSSC(y)) ** 2)


class TI_Loss(nn.Module):
    """
    References: https://github.com/TopoXLab/TopoInteraction

    The proposed topological interaction (TI) module encodes topological interactions by computing the critical voxels map.
    The critical voxels map contains the locations which induce errors in the topological interactions.
    The TI loss is introduced based on the topological interaction module.
    """

    def __init__(self, dim, connectivity, inclusion, exclusion, min_thick=1):
        """
        :param dim: 2 if 2D; 3 if 3D
        :param connectivity: 4 or 8 for 2D; 6 or 26 for 3D
        :param inclusion: list of [A,B] classes where A is completely surrounded by B.
        :param exclusion: list of [A,C] classes where A and C exclude each other.
        :param min_thick: Minimum thickness/separation between the two classes. Only used if connectivity is 8 for 2D or 26 for 3D
        """
        super(TI_Loss, self).__init__()

        self.dim = dim
        self.connectivity = connectivity
        self.min_thick = min_thick
        self.interaction_list = []
        self.sum_dim_list = None
        self.conv_op = None
        self.apply_nonlin = lambda x: torch.nn.functional.softmax(x, 1)
        self.ce_loss_func = torch.nn.CrossEntropyLoss(reduction='none')

        if self.dim == 2:
            self.sum_dim_list = [1, 2, 3]
            self.conv_op = torch.nn.functional.conv2d
        elif self.dim == 3:
            self.sum_dim_list = [1, 2, 3, 4]
            self.conv_op = torch.nn.functional.conv3d

        self.set_kernel()

        if len(inclusion) != 0:
            for inc in inclusion:
                temp_pair = []
                temp_pair.append(True)  # type inclusion
                temp_pair.append(inc[0])
                temp_pair.append(inc[1])
                self.interaction_list.append(temp_pair)

        if len(exclusion) != 0:
            for exc in exclusion:
                temp_pair = []
                temp_pair.append(False)  # type exclusion
                temp_pair.append(exc[0])
                temp_pair.append(exc[1])
                self.interaction_list.append(temp_pair)
        else:
            raise ValueError(f"Prams 'inclusion' or 'exclusion' must have values")

    def set_kernel(self):
        """
        Sets the connectivity kernel based on user's sepcification of dim, connectivity, min_thick
        """
        k = 2 * self.min_thick + 1
        if self.dim == 2:
            if self.connectivity == 4:
                np_kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
            elif self.connectivity == 8:
                np_kernel = np.ones((k, k))

        elif self.dim == 3:
            if self.connectivity == 6:
                np_kernel = np.array([
                    [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                    [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
                    [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
                ])
            elif self.connectivity == 26:
                np_kernel = np.ones((k, k, k))

        self.kernel = torch_kernel = torch.from_numpy(np.expand_dims(np.expand_dims(np_kernel, axis=0), axis=0))

    def topological_interaction_module(self, P):
        """
        Given a discrete segmentation map and the intended topological interactions, this module computes the critical voxels map.
        :param P: Discrete segmentation map
        :return: Critical voxels map
        """

        for ind, interaction in enumerate(self.interaction_list):
            interaction_type = interaction[0]
            label_A = interaction[1]
            label_C = interaction[2]

            # Get Masks
            mask_A = torch.where(P == label_A, 1.0, 0.0).double()
            if interaction_type:
                mask_C = torch.where(P == label_C, 1.0, 0.0).double()
                mask_C = torch.logical_or(mask_C, mask_A).double()
                mask_C = torch.logical_not(mask_C).double()
            else:
                mask_C = torch.where(P == label_C, 1.0, 0.0).double()

            # Get Neighbourhood Information
            # padding='same' needs pytorch >= 1.9, it means the output shape keep the same with input.
            # When kernel size is 3, padding=1 got the same perform.
            neighbourhood_C = self.conv_op(mask_C, self.kernel.double(), padding=1)
            neighbourhood_C = torch.where(neighbourhood_C >= 1.0, 1.0, 0.0)
            neighbourhood_A = self.conv_op(mask_A, self.kernel.double(), padding=1)
            neighbourhood_A = torch.where(neighbourhood_A >= 1.0, 1.0, 0.0)

            # Get the pixels which induce errors
            violating_A = neighbourhood_C * mask_A
            violating_C = neighbourhood_A * mask_C
            violating = violating_A + violating_C
            violating = torch.where(violating >= 1.0, 1.0, 0.0)

            if ind == 0:
                critical_voxels_map = violating
            else:
                critical_voxels_map = torch.logical_or(critical_voxels_map, violating).double()

        return critical_voxels_map

    def forward(self, x, y):
        """
        The forward function computes the TI loss value.
        :param x: Likelihood map of shape: b, c, x, y(, z) with c = total number of classes
        :param y: GT of shape: b, c, x, y(, z) with c=1. The GT should only contain values in [0,L) range where L is the total number of classes.
        :return:  TI loss value
        """

        if x.device.type == "cuda":
            self.kernel = self.kernel.cuda(x.device.index)

        # Obtain discrete segmentation map
        x_softmax = self.apply_nonlin(x)
        P = torch.argmax(x_softmax, dim=1)
        P = torch.unsqueeze(P.double(), dim=1)
        del x_softmax

        # Call the Topological Interaction Module
        critical_voxels_map = self.topological_interaction_module(P)

        # Compute the TI loss value
        ce_tensor = torch.unsqueeze(self.ce_loss_func(x.double(), y[:, 0].long()), dim=1)
        ce_tensor[:, 0] = ce_tensor[:, 0] * torch.squeeze(critical_voxels_map, dim=1)
        ce_loss_value = ce_tensor.sum(dim=self.sum_dim_list).mean()

        return ce_loss_value


if __name__ == "__main__":
    """
    Sample usage. In order to test the code, Input and GT are randomly populated with values.
    Set the dim (2 for 2D; 3 for 3D) correctly to run relevant code.

    The samples provided enforce the following interactions:
        Enforce class 1 to be completely surrounded by class 2
        Enforce class 2 to be excluded from class 3
        Enforce class 3 to be excluded from class 4
    """

    # Parameters for creating random input
    num_classes = height = width = depth = 5

    dim = 3

    if dim == 2:
        x = torch.rand(1, num_classes, height, width)
        y = torch.randint(0, num_classes, (1, 1, height, width))

        ti_loss_weight = 1e-4
        ti_loss_func = TI_Loss(dim=2, connectivity=4, inclusion=[[1, 2]], exclusion=[[2, 3], [3, 4]])
        ti_loss_value = ti_loss_func(x, y) if ti_loss_weight != 0 else 0
        ti_loss_value = ti_loss_weight * ti_loss_value
        print("ti_loss_value: ", ti_loss_value)

    elif dim == 3:
        x = torch.rand(1, num_classes, depth, height, width)
        y = torch.randint(0, num_classes, (1, 1, depth, height, width))

        ti_loss_weight = 1e-6
        ti_loss_func = TI_Loss(dim=3, connectivity=26, inclusion=[[1, 2]], exclusion=[[2, 3], [3, 4]], min_thick=1)
        ti_loss_value = ti_loss_func(x, y) if ti_loss_weight != 0 else 0
        ti_loss_value = ti_loss_weight * ti_loss_value
        print("ti_loss_value: ", ti_loss_value)
