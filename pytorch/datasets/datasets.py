import math
import sys
sys.path.append(r'F:\shb_src\from_github\AIforMS\registration-in-action\pytorch')

import os
import numpy as np
from PIL import Image
import nibabel as nib
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import datasets, transforms

from utils import ImgTransform


class LabelSampler(Sampler):
    def __init__(self, mask, for_what: str = 'train', val_size: int = 512):
        if for_what not in ['train', 'val', 'test']:
            raise ValueError(f"'for_what' need to be one of ['train', 'val', 'test'], but go {for_what}")
        for_what = for_what

        if not isinstance(mask, torch.Tensor):
            mask = torch.tensor(mask)

        if for_what == 'val':
            # 验证集从训练集中瓜分
            indices = torch.nonzero(mask)[:val_size]
        elif for_what == 'train':
            indices = torch.nonzero(mask)[val_size:]
        elif for_what == 'test':
            indices = torch.nonzero(mask)

        self.indices = indices.flatten().numpy()

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return self.indices.shape[0]


class MNISTDataset(Dataset):
    def __init__(self,
                 root,
                 for_what: str = 'train',
                 choose_label=5,
                 val_size=512):

        self.for_what = for_what

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Pad(2),  # 28x28 -> 32x32
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        train_set = datasets.MNIST(
            root=root,
            train=True,
            download=True
        )
        test_set = datasets.MNIST(
            root=root,
            train=False,
            download=True
        )

        train_val_indices = torch.where(train_set.train_labels == choose_label)[0]
        test_indices = torch.where(test_set.test_labels == choose_label)[0]

        val_indices = np.random.choice(train_val_indices, val_size)  # 随机选择验证集
        train_indices = torch.tensor([i for i in train_val_indices.numpy() if i not in val_indices])

        self.train_imgs = train_set.data[train_indices]
        self.train_labels = train_set.train_labels[train_indices]

        self.val_imgs = train_set.data[val_indices]
        self.val_labels = train_set.train_labels[val_indices]

        self.test_imgs = test_set.data[test_indices]
        self.test_labels = test_set.test_labels[test_indices]

        if self.for_what == 'train':
            self.len_ = self.train_labels.shape[0]
        if self.for_what == 'val':
            self.len_ = self.val_labels.shape[0]
        if self.for_what == 'test':
            self.len_ = self.test_labels.shape[0]

    def __len__(self):
        return self.len_

    def __getitem__(self, idx):
        fix_idx = np.random.choice(np.where(np.arange(self.len_) != idx)[0])  # 随机选取一个 fixed image

        if self.for_what == 'train':
            moving_img, fixed_img = self.train_imgs[idx], self.train_imgs[fix_idx]
            # convert tensor imgs to PIL imgs for transforms
            moving_img = Image.fromarray(moving_img.numpy(), mode='L')
            fixed_img = Image.fromarray(fixed_img.numpy(), mode='L')
            # transforms
            moving_img = self.transform(moving_img)
            fixed_img = self.transform(fixed_img)
            return moving_img, fixed_img

        if self.for_what == 'val':
            moving_img, fixed_img = self.val_imgs[idx], self.val_imgs[fix_idx]
            # convert tensor imgs to PIL imgs for transforms
            moving_img = Image.fromarray(moving_img.numpy(), mode='L')
            fixed_img = Image.fromarray(fixed_img.numpy(), mode='L')
            # transforms
            moving_img = self.transform(moving_img)
            fixed_img = self.transform(fixed_img)
            return moving_img, fixed_img

        if self.for_what == 'test':
            moving_img, fixed_img = self.test_imgs[idx], self.test_imgs[fix_idx]
            # convert tensor imgs to PIL imgs for transforms
            moving_img = Image.fromarray(moving_img.numpy(), mode='L')
            fixed_img = Image.fromarray(fixed_img.numpy(), mode='L')
            # transforms
            moving_img = self.transform(moving_img)
            fixed_img = self.transform(fixed_img)
            return moving_img, fixed_img


def mnist(root='datasets/data/mnist',
          for_what='train',
          choose_label=5,
          val_size=512,
          batch_size=64,
          num_workers=2):
    """
    MNIST 数据加载器
    :param root:
    :param choose_label: 选择指定的数字用于配准训练
    :param batch_size:
    :param val_size: 验证集的数据量
    :return: data_loader: moving_img, fixed_img
    """
    mnist_dataset = MNISTDataset(root=root, for_what=for_what, choose_label=choose_label, val_size=val_size)
    data_loader = DataLoader(mnist_dataset, batch_size=batch_size, num_workers=num_workers)

    return data_loader


class LPBADataset(Dataset):
    def __init__(self,
                 image_folder,
                 image_name,
                 label_folder,
                 label_name,
                 scannumbers):

        if image_name.find("?") == -1 or label_name.find("?") == -1:
            raise ValueError('error! filename must contain \"?\" to insert your chosen numbers')

        self.img_paths = []
        self.label_paths = []

        for i in scannumbers:
            self.img_paths.append(os.path.join(image_folder, image_name.replace("?", str(i))))
            self.label_paths.append(os.path.join(label_folder, label_name.replace("?", str(i))))

        self.len_ = min(len(self.img_paths), len(self.label_paths))

    def __len__(self):
        return self.len_

    def __getitem__(self, idx):
        fix_idx = np.random.choice(np.where(np.arange(self.len_) != idx)[0])  # 随机选取一个 fixed image

        # 用 nibabel 读取的图像会被旋转，需要得到原图的 affine 才能还原，很迷。但是在 inference 的时候保存的图像又是正常的，不不知为何。
        # 可能是原本网络的输出就是旋转过后的，再次用 nibabel 保存之后又转回来了？
        moving_img_path = self.img_paths[idx]
        moving_label_path = self.label_paths[idx]
        fixed_img_path = self.img_paths[fix_idx]
        fixed_label_path = self.label_paths[fix_idx]

        moving_imgs = sitk.GetArrayFromImage(sitk.ReadImage(moving_img_path))[np.newaxis, ...].astype(np.float32)
        moving_labels = sitk.GetArrayFromImage(sitk.ReadImage(moving_label_path))[np.newaxis, ...].astype(np.float32)

        fixed_imgs = sitk.GetArrayFromImage(sitk.ReadImage(fixed_img_path))[np.newaxis, ...].astype(np.float32)
        fixed_labels = sitk.GetArrayFromImage(sitk.ReadImage(fixed_label_path))[np.newaxis, ...].astype(np.float32)

        # # 这两个标签没有对应的结构，已经预处理
        # label_arr[label_arr == 181.] = 0.
        # label_arr[label_arr == 182.] = 0.
        return moving_imgs, moving_labels, fixed_imgs, fixed_labels

    def get_labels_num(self):
        a_label = nib.load(self.label_paths[0]).get_fdata()
        return int(len(np.unique(a_label)))


def lpba(logger,
         img_folder=None,
         img_name=None,
         label_folder=None,
         label_name=None,
         train_scannumbers=None,
         val_scannumbers=None,
         batch_size=2,
         is_shuffle=True,
         num_workers=2):

    train_dataset = LPBADataset(image_folder=img_folder,
                                image_name=img_name,
                                label_folder=label_folder,
                                label_name=label_name,
                                scannumbers=train_scannumbers)

    val_dataset = LPBADataset(image_folder=img_folder,
                              image_name=img_name,
                              label_folder=label_folder,
                              label_name=label_name,
                              scannumbers=val_scannumbers)

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=is_shuffle,
                              drop_last=True,
                              num_workers=num_workers)

    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=1, num_workers=num_workers)

    num_labels = train_dataset.get_labels_num()
    logger.info(f'Training set sizes: {len(train_dataset)}, Train loader size: {len(train_loader)}, '
                f'Validation set sizes: {len(val_dataset)}')

    return train_loader, val_loader, num_labels


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import visdom

    vis = visdom.Visdom()

    batch_size = 4
    row = int(math.sqrt(batch_size))  # 下取整
    col = math.ceil(batch_size / row)  # 上取整

    val_loader = mnist(choose_label=5, batch_size=batch_size, for_what='val')
    moving_img, fixed_img = next(iter(val_loader))

    # data_loader = mnist(choose_label=5, batch_size=batch_size, for_what='test')
    # for i, (moving_img, fixed_img) in enumerate(data_loader):
    #     imgs = moving_img.numpy().transpose((0, 2, 3, 1))  # (N, H, W, C), 把通道换到最后一维
    #     for i in range(batch_size):
    #         plt.subplot(row, col, i + 1)
    #         plt.tight_layout()
    #         plt.imshow(imgs[i], cmap='gray')  # interpolation='none'
    #         plt.xticks([])
    #         plt.yticks([])
    #     plt.show()
    #     break

    trans = ImgTransform(scale_type='max-min')
    moving_img = (trans(moving_img) * 255).type(dtype=torch.uint8)
    fixed_img = (trans(fixed_img) * 255).type(dtype=torch.uint8)

    vis.images(moving_img, nrow=1, win='mov', opts={'title': 'Moving Images'})
    vis.images(moving_img, nrow=1, win='moved', opts={'title': 'Moved Images'})
    vis.images(fixed_img, nrow=1, win='fix', opts={'title': 'fixed Images'})
