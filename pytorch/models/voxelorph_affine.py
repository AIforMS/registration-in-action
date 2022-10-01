import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

import inspect
import functools


def default_unet_features():
    nb_features = [
        [16, 32, 32, 32],  # encoder
        [32, 32, 32, 32, 32, 16, 16]  # decoder
    ]
    return nb_features


def store_config_args(func):
    """
    Class-method decorator that saves every argument provided to the
    function as a dictionary in 'self.config'. This is used to assist
    model loading - see LoadableModel.
    """

    attrs, varargs, varkw, defaults = inspect.getargspec(func)

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        self.config = {}

        # first save the default values
        if defaults:
            for attr, val in zip(reversed(attrs), reversed(defaults)):
                self.config[attr] = val

        # next handle positional args
        for attr, val in zip(attrs[1:], args):
            self.config[attr] = val

        # lastly handle keyword args
        if kwargs:
            for attr, val in kwargs.items():
                self.config[attr] = val

        return func(self, *args, **kwargs)

    return wrapper


class LoadableModel(nn.Module):
    """
    Base class for easy pytorch model loading without having to manually
    specify the architecture configuration at load time.

    We can cache the arguments used to the construct the initial network, so that
    we can construct the exact same network when loading from file. The arguments
    provided to __init__ are automatically saved into the object (in self.config)
    if the __init__ method is decorated with the @store_config_args utility.
    """

    # this constructor just functions as a check to make sure that every
    # LoadableModel subclass has provided an internal config parameter
    # either manually or via store_config_args
    def __init__(self, *args, **kwargs):
        if not hasattr(self, 'config'):
            raise RuntimeError('models that inherit from LoadableModel must decorate the '
                               'constructor with @store_config_args')
        super().__init__(*args, **kwargs)

    def save(self, path):
        """
        Saves the model configuration and weights to a pytorch file.
        """
        # don't save the transformer_grid buffers - see SpatialTransformer doc for more info
        sd = self.state_dict().copy()
        grid_buffers = [key for key in sd.keys() if key.endswith('.grid')]
        for key in grid_buffers:
            sd.pop(key)
        torch.save({'config': self.config, 'model_state': sd}, path)

    @classmethod
    def load(cls, path, device='cuda'):
        """
        Load a python model configuration and weights.
        """
        checkpoint = torch.load(path, map_location=torch.device(device))
        model = cls(**checkpoint['config'])
        model.load_state_dict(checkpoint['model_state'], strict=False)
        return model


class Mlp(nn.Module):
    def __init__(self, in_channels, hidden_features=None, out_channels=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_channels = out_channels or in_channels
        hidden_features = hidden_features or in_channels
        self.fc1 = nn.Linear(in_channels, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_channels)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class VxmAffineNet(nn.Module):
    def __init__(self,
                 inshape,
                 nb_unet_features=None,
                 nb_unet_levels=None,
                 unet_feat_mult=1,
                 nb_unet_conv_per_level=1,
                 src_feats=1,
                 trg_feats=1,
                 unet_half_res=False,
                 norm_layer=nn.LayerNorm,):
        """
        :param inshape: Input shape. e.g. (192, 192, 192)
        :param nb_unet_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. If None (default),
                the unet features are defined by the default config described in the unet class documentation.
        :param nb_unet_levels: Number of levels in unet. Only used when nb_features is an integer. Default is None.
        :param unet_feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. Default is 1.
        :param nb_unet_conv_per_level: Number of convolutions per unet level. Default is 1.
                int_steps: Number of flow integration steps. The warp is non-diffeomorphic when this value is 0.
        :param src_feats: Number of source image features. Default is 1.
        :param trg_feats: Number of target image features. Default is 1.
        :param unet_half_res: Skip the last unet decoder upsampling. Requires that int_downsize=2. Default is False.
        :param norm_layer: Layer norm for Mlp.
        """
        super(VxmAffineNet, self).__init__()

        # ensure correct dimensionality
        self.ndims = len(inshape)
        assert self.ndims in [2, 3], 'Only support 2D and 3D, ndims should be one of 2 or 3. found: %d' % self.ndims

        if self.ndims == 2:
            out_shear_channels = 2
        elif self.ndims == 3:
            out_shear_channels = 6

        # configure core unet model
        self.unet_model = Unet(
            inshape,
            infeats=(src_feats + trg_feats),
            nb_features=nb_unet_features,
            nb_levels=nb_unet_levels,
            feat_mult=unet_feat_mult,
            nb_conv_per_level=nb_unet_conv_per_level,
            half_res=unet_half_res)

        self.gap = getattr(F, f"adaptive_avg_pool{self.ndims}d")

        mlp_in_channels = self.unet_model.final_nf
        hidden_channels = mlp_in_channels // 2
        out_channels = hidden_channels // 2
        channels_lst = [mlp_in_channels, hidden_channels, out_channels]

        self.norm = norm_layer(inshape)
        self.mlp = Mlp(channels_lst[0], channels_lst[1], channels_lst[2])

        self.aff_head = nn.Linear(channels_lst[2], self.ndims)
        self.aff_head.weight = nn.Parameter(torch.zeros(self.aff_head.weight.shape))
        self.aff_head.bias = nn.Parameter(torch.zeros(self.aff_head.bias.shape))

        self.scl_head = nn.Linear(channels_lst[2], self.ndims)
        self.scl_head.weight = nn.Parameter(torch.zeros(self.scl_head.weight.shape))
        self.scl_head.bias = nn.Parameter(torch.zeros(self.scl_head.bias.shape))

        self.trans_head = nn.Linear(channels_lst[2], self.ndims)
        self.trans_head.weight = nn.Parameter(torch.zeros(self.trans_head.weight.shape))
        self.trans_head.bias = nn.Parameter(torch.zeros(self.trans_head.bias.shape))

        self.shear_head = nn.Linear(channels_lst[2], out_shear_channels)
        self.shear_head.weight = nn.Parameter(torch.zeros(self.shear_head.weight.shape))
        self.shear_head.bias = nn.Parameter(torch.zeros(self.shear_head.bias.shape))

        self.affine_trans = AffineTransformer()

    def forward(self, source, target):
        N = source.shape[0]
        # concatenate inputs and propagate unet
        x = torch.cat([source, target], dim=1)
        x = self.unet_model(x)

        x = self.norm(x)
        x = self.gap(x, 1)
        x = x.reshape(N, self.unet_model.final_nf)

        x = self.mlp(x)
        aff = self.aff_head(x)
        scl = self.scl_head(x)
        trans = self.trans_head(x)
        shr = self.shear_head(x)

        out, mat = self.affine_trans(source, aff, scl, trans, shr)
        return out, mat


class AffineTransformer(nn.Module):
    """
    3-D Affine Transformer
    """

    def __init__(self, mode='bilinear'):
        super().__init__()
        self.mode = mode

    def forward(self, src, affine, scale, translate, shear):
        if src.shape == 5:
            # 3D 放射配准
            theta_x = affine[:, 0]
            theta_y = affine[:, 1]
            theta_z = affine[:, 2]

            scale_x = scale[:, 0]
            scale_y = scale[:, 1]
            scale_z = scale[:, 2]

            trans_x = translate[:, 0]
            trans_y = translate[:, 1]
            trans_z = translate[:, 2]

            shear_xy = shear[:, 0]
            shear_xz = shear[:, 1]
            shear_yx = shear[:, 2]
            shear_yz = shear[:, 3]
            shear_zx = shear[:, 4]
            shear_zy = shear[:, 5]

            rot_mat_x = torch.stack(
                [torch.stack([torch.ones_like(theta_x), torch.zeros_like(theta_x), torch.zeros_like(theta_x)], dim=1),
                 torch.stack([torch.zeros_like(theta_x), torch.cos(theta_x), -torch.sin(theta_x)], dim=1),
                 torch.stack([torch.zeros_like(theta_x), torch.sin(theta_x), torch.cos(theta_x)], dim=1)], dim=2).cuda()

            rot_mat_y = torch.stack(
                [torch.stack([torch.cos(theta_y), torch.zeros_like(theta_y), torch.sin(theta_y)], dim=1),
                 torch.stack([torch.zeros_like(theta_y), torch.ones_like(theta_x), torch.zeros_like(theta_x)], dim=1),
                 torch.stack([-torch.sin(theta_y), torch.zeros_like(theta_y), torch.cos(theta_y)], dim=1)], dim=2).cuda()

            rot_mat_z = torch.stack(
                [torch.stack([torch.cos(theta_z), -torch.sin(theta_z), torch.zeros_like(theta_y)], dim=1),
                 torch.stack([torch.sin(theta_z), torch.cos(theta_z), torch.zeros_like(theta_y)], dim=1),
                 torch.stack([torch.zeros_like(theta_y), torch.zeros_like(theta_y), torch.ones_like(theta_x)], dim=1)],
                dim=2).cuda()

            scale_mat = torch.stack(
                [torch.stack([scale_x, torch.zeros_like(theta_z), torch.zeros_like(theta_y)], dim=1),
                 torch.stack([torch.zeros_like(theta_z), scale_y, torch.zeros_like(theta_y)], dim=1),
                 torch.stack([torch.zeros_like(theta_y), torch.zeros_like(theta_y), scale_z], dim=1)], dim=2).cuda()

            shear_mat = torch.stack(
                [torch.stack([torch.ones_like(theta_x), torch.tan(shear_xy), torch.tan(shear_xz)], dim=1),
                 torch.stack([torch.tan(shear_yx), torch.ones_like(theta_x), torch.tan(shear_yz)], dim=1),
                 torch.stack([torch.tan(shear_zx), torch.tan(shear_zy), torch.ones_like(theta_x)], dim=1)], dim=2).cuda()

            trans = torch.stack([trans_x, trans_y, trans_z], dim=1).unsqueeze(dim=2)
            mat = torch.bmm(shear_mat, torch.bmm(scale_mat, torch.bmm(rot_mat_z, torch.matmul(rot_mat_y, rot_mat_x))))
            mat = torch.cat([mat, trans], dim=-1)
            grid = F.affine_grid(mat, [src.shape[0], 3, src.shape[2], src.shape[3], src.shape[4]], align_corners=True)

            return F.grid_sample(src, grid, align_corners=True, mode=self.mode), mat
        elif src.shape == 4:
            # 2D 放射配准
            theta_x = affine[:, 0]
            theta_y = affine[:, 1]

            scale_x = scale[:, 0]
            scale_y = scale[:, 1]

            trans_x = translate[:, 0]
            trans_y = translate[:, 1]

            shear_xy = shear[:, 0]
            shear_yx = shear[:, 2]

            rot_mat_x = torch.stack(
                [torch.stack([torch.cos(theta_x), -torch.sin(theta_x), torch.zeros_like(theta_x)], dim=1),
                 torch.stack([torch.sin(theta_x), torch.cos(theta_x),  torch.zeros_like(theta_x)], dim=1)], dim=2).cuda()

            rot_mat_y = torch.stack(
                [torch.stack([torch.cos(theta_y), torch.zeros_like(theta_y), torch.sin(theta_y)], dim=1),
                 torch.stack([torch.zeros_like(theta_y), torch.ones_like(theta_x), torch.zeros_like(theta_x)], dim=1),
                 torch.stack([-torch.sin(theta_y), torch.zeros_like(theta_y), torch.cos(theta_y)], dim=1)],
                dim=2).cuda()

            scale_mat = torch.stack(
                [torch.stack([scale_x, torch.zeros_like(theta_z), torch.zeros_like(theta_y)], dim=1),
                 torch.stack([torch.zeros_like(theta_z), scale_y, torch.zeros_like(theta_y)], dim=1),
                 torch.stack([torch.zeros_like(theta_y), torch.zeros_like(theta_y), scale_z], dim=1)], dim=2).cuda()

            shear_mat = torch.stack(
                [torch.stack([torch.ones_like(theta_x), torch.tan(shear_xy), torch.tan(shear_xz)], dim=1),
                 torch.stack([torch.tan(shear_yx), torch.ones_like(theta_x), torch.tan(shear_yz)], dim=1),
                 torch.stack([torch.tan(shear_zx), torch.tan(shear_zy), torch.ones_like(theta_x)], dim=1)],
                dim=2).cuda()

            trans = torch.stack([trans_x, trans_y, trans_z], dim=1).unsqueeze(dim=2)
            mat = torch.bmm(shear_mat, torch.bmm(scale_mat, torch.bmm(rot_mat_z, torch.matmul(rot_mat_y, rot_mat_x))))
            mat = torch.cat([mat, trans], dim=-1)
            grid = F.affine_grid(mat, [src.shape[0], 3, src.shape[2], src.shape[3], src.shape[4]], align_corners=True)

            return F.grid_sample(src, grid, align_corners=True, mode=self.mode), mat


class Unet(nn.Module):
    """
    A unet architecture. Layer features can be specified directly as a list of encoder and decoder
    features or as a single integer along with a number of unet levels. The default network features
    per layer (when no options are specified) are:

        encoder: [16, 32, 32, 32]
        decoder: [32, 32, 32, 32, 32, 16, 16]
    """

    def __init__(self,
                 inshape=None,
                 infeats=None,
                 nb_features=None,
                 nb_levels=None,
                 max_pool=2,
                 feat_mult=1,
                 nb_conv_per_level=1,
                 half_res=False):
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            infeats: Number of input features.
            nb_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer.
                If None (default), the unet features are defined by the default config described in
                the class documentation.
            nb_levels: Number of levels in unet. Only used when nb_features is an integer.
                Default is None.
            feat_mult: Per-level feature multiplier. Only used when nb_features is an integer.
                Default is 1.
            nb_conv_per_level: Number of convolutions per unet level. Default is 1.
            half_res: Skip the last decoder upsampling. Default is False.
        """

        super().__init__()

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # cache some parameters
        self.half_res = half_res

        # default encoder and decoder layer features if nothing provided
        if nb_features is None:
            nb_features = default_unet_features()

        # build feature list automatically
        if isinstance(nb_features, int):
            if nb_levels is None:
                raise ValueError('must provide unet nb_levels if nb_features is an integer')
            feats = np.round(nb_features * feat_mult ** np.arange(nb_levels)).astype(int)
            nb_features = [
                np.repeat(feats[:-1], nb_conv_per_level),
                np.repeat(np.flip(feats), nb_conv_per_level)
            ]
        elif nb_levels is not None:
            raise ValueError('cannot use nb_levels if nb_features is not an integer')

        # extract any surplus (full resolution) decoder convolutions
        enc_nf, dec_nf = nb_features
        nb_dec_convs = len(enc_nf)
        final_convs = dec_nf[nb_dec_convs:]
        dec_nf = dec_nf[:nb_dec_convs]
        self.nb_levels = int(nb_dec_convs / nb_conv_per_level) + 1

        if isinstance(max_pool, int):
            max_pool = [max_pool] * self.nb_levels

        # cache downsampling / upsampling operations
        MaxPooling = getattr(nn, 'MaxPool%dd' % ndims)
        self.pooling = [MaxPooling(s) for s in max_pool]
        self.upsampling = [nn.Upsample(scale_factor=s, mode='nearest') for s in max_pool]

        # configure encoder (down-sampling path)
        prev_nf = infeats
        encoder_nfs = [prev_nf]
        self.encoder = nn.ModuleList()
        for level in range(self.nb_levels - 1):
            convs = nn.ModuleList()
            for conv in range(nb_conv_per_level):
                nf = enc_nf[level * nb_conv_per_level + conv]
                convs.append(ConvBlock(ndims, prev_nf, nf))
                prev_nf = nf
            self.encoder.append(convs)
            encoder_nfs.append(prev_nf)

        # configure decoder (up-sampling path)
        encoder_nfs = np.flip(encoder_nfs)
        self.decoder = nn.ModuleList()
        for level in range(self.nb_levels - 1):
            convs = nn.ModuleList()
            for conv in range(nb_conv_per_level):
                nf = dec_nf[level * nb_conv_per_level + conv]
                convs.append(ConvBlock(ndims, prev_nf, nf))
                prev_nf = nf
            self.decoder.append(convs)
            if not half_res or level < (self.nb_levels - 2):
                prev_nf += encoder_nfs[level]

        # now we take care of any remaining convolutions
        self.remaining = nn.ModuleList()
        for num, nf in enumerate(final_convs):
            self.remaining.append(ConvBlock(ndims, prev_nf, nf))
            prev_nf = nf

        # cache final number of features
        self.final_nf = prev_nf

    def forward(self, x):

        # encoder forward pass
        x_history = [x]
        for level, convs in enumerate(self.encoder):
            for conv in convs:
                x = conv(x)
            x_history.append(x)
            x = self.pooling[level](x)

        # decoder forward pass with upsampling and concatenation
        for level, convs in enumerate(self.decoder):
            for conv in convs:
                x = conv(x)
            if not self.half_res or level < (self.nb_levels - 2):
                x = self.upsampling[level](x)
                x = torch.cat([x, x_history.pop()], dim=1)

        # remaining convs at full resolution
        for conv in self.remaining:
            x = conv(x)

        return x


class ConvBlock(nn.Module):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """

    def __init__(self, ndims, in_channels, out_channels, stride=1):
        super().__init__()

        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.main = Conv(in_channels, out_channels, 3, stride, 1)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        out = self.main(x)
        out = self.activation(out)
        return out


class ApplyAffine(nn.Module):
    """
    Affine Transformer
    """
    def __init__(self, ):
        super().__init__()

    def forward(self, src, mat, mode='bilinear'):
        grid = F.affine_grid(mat, src.shape, align_corners=True)
        return F.grid_sample(src, grid, align_corners=True, mode=mode)


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    H, W, D = 32, 32, 32  # 80, 80, 80

    shape = [H, W, D]

    x1 = torch.randn([1, 1, *shape], device=device)
    x2 = torch.randn([1, 1, *shape], device=device)

    enc_nf = [16, 32, 32, 32]
    dec_nf = [32, 32, 32, 32, 32, 16, 16]
    net = VxmAffineNet(
        inshape=shape,
        nb_unet_features=[enc_nf, dec_nf],
        src_feats=1,
        trg_feats=1)
    net.to(device)
    net.train()

    affiner = ApplyAffine()

    out1, mat = net(x1, x2)

    print(f"out: {out1.shape}, mat: {mat.shape}")
    # y_source: torch.Size([1, 2, 192, 160, 192]), pos_flow: torch.Size([1, 3, 192, 160, 192])

    out2 = affiner(x1, mat)

    assert (out1 == out2).all(), "damn it"
