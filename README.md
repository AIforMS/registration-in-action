# registration-in-action

深度学习图像配准（Deep Learning Image Registration, DLIR），基于 [VoxelMorph](https://github.com/voxelmorph/voxelmorph) 。

欢迎补充

---

- 推荐 DLIR 入门教程：

    - [Deep learning in MRI](https://theaisummer.com/mri-beyond-segmentation/)
    - [Deep learning and medical imaging](https://theaisummer.com/medical-image-deep-learning/)
    - [learn2reg voxelmorph](https://www.kaggle.com/code/adalca/learn2reg/notebook)
    - [DeepReg Intro to DLIR](https://github.com/DeepRegNet/DeepReg/blob/main/docs/Intro_to_Medical_Image_Registration.ipynb)


- 推荐 DLIR 名人堂：

    | 学者 | 代表作 | 个人网页 | 备注 |
    | -- | -- | -- | -- |
    | Adrian V. Dalca | [VoxelMorph](https://arxiv.org/abs/1809.05231) | http://www.mit.edu/~adalca/ | VM 掀起 DLIR 热潮 |
    | MP Heinrich, L Hansen | [Deeds](https://github.com/mattiaspaul/deedsBCV)、[OBELISK](https://github.com/mattiaspaul/OBELISK)、[PDD_Net](https://github.com/multimodallearning/pdd_net) | [mpheinrich](http://mpheinrich.de/)、 [researchgate](https://www.researchgate.net/profile/Mattias-Heinrich) | Deeds 是我用过配准性能最好的，幸好它是传统配准 :grimacing: ，OBELISK 分割性能也很好，而且轻量。 |
    | Yipeng Hu | [Weakly Reg](https://www.sciencedirect.com/science/article/pii/S1361841518301051?via%3Dihub)、[DeepReg](https://github.com/DeepRegNet/DeepReg) | https://iris.ucl.ac.uk/iris/browse/profile?upi=YHUXX66 | 我们开始使用分割标签做弱监督配准学习 |
    | Tony C.W. Mok | [LapIRN](https://github.com/cwmok)、[C2FViT](https://github.com/cwmok/C2FViT) | https://cwmok.github.io/ | 无监督脑部配准，他总是第一名 |
    | Chen Junyu | [TransMorph](https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration)、[ViT-V-Net](https://github.com/junyuchen245/ViT-V-Net_for_3D_Image_Registration_Pytorch) | https://github.com/junyuchen245 | Transformer + CNN 版的 VoxelMorph，很多代码可以借鉴 |  
    | ANTs | [Advanced Normalization Tools](https://github.com/ANTsX/ANTsPy) | [GitHub](https://github.com/ANTsX/ANTsPy)、[CSDN](https://blog.csdn.net/zuzhiang/article/details/104930000) | 常用的传统配准方法 baseline |

- 推荐文献检索网站：

    - https://www.semanticscholar.org/
    - https://www.connectedpapers.com/
    - https://paperswithcode.com/

- 推荐综述文献：

    - Transforming medical imaging with Transformers? A comparative review of key properties, current progresses, and future perspectives. Medical Image Analysis, 2022.
    - Deep learning in medical image registration: a survey. Machine Vision and Applications, 2019.
    - Is Image-to-Image Translation the Panacea for Multimodal Image Registration? A Comparative Study. CVPR, 2021.

## 2D 配准

### 数据集

MNIST

### 训练

> **train_vm_2d.py**

本地运行：

- 启动 visdom 用于实时可视化

```shell
python -m visdom.server
```

- 启动训练

```shell
python train_vm_2d.py \
    -output output/mnist/ \
    -is_visdom True \
    -choose_label 5 \
    -val_interval 1 \
    -save_interval 50
```

数字 5 的预训练结果见 `ckpts/mnist`

![visdom-train](./pytorch/ckpts/mnist/visdom-train.jpg)

### 预测

> **register_vm_2d.py**

本地运行：

```shell
python register_vm_2d.py \
    -output output/mnist_test/ \
    -model output/mnist/reg_net_299.pth \
    -is_visdom True \
    -choose_label 5 \
```

预测结果可视化：

![visdom-train](./pytorch/ckpts/mnist/visdom-test-5.jpg)

模型神奇地显示出了泛化效果，在没有参与训练的数字 3 上也可以配准， 其它数字也一样：

![visdom-train](./pytorch/ckpts/mnist/visdom-test-3.jpg)
![visdom-train](./pytorch/ckpts/mnist/visdom-test-7.jpg)


## 3D 配准

### 数据集

[LPBA40](https://github.com/AIforMS/seg-with-ti/releases/tag/v0.1.2) - 3D 脑部 MRI

### 训练

> **train_vm_3d.py**

训练脚本：

```shell
#!/bin/bash
#SBATCH -J LPBA
#SBATCH -p gpu1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -o logs/trainOutLPBA.txt
#SBATCH -e logs/trainErrLPBA.txt

SLURM_SUBMIT_DIR=/public/home/jd_shb/fromgithub/registration-in-action
cd $SLURM_SUBMIT_DIR
NP=$SLURM_JOB_GPUS

CUDA_VISIBLE_DEVICES=$NP python train_vm_3d.py \
    -dataset lpba40 \
    -output output/lpba/ \
    -resume /public/home/jd_shb/fromgithub/registration-in-action/output/lpba/lpba40_best71.pth \
    -batch_size 2 \
    -lr 1e-4 \
    -apply_lr_scheduler \
    -epochs 800 \
    -weakly_sup \  # 使用分割标签辅助训练
    -sim_loss MSE \
    -sim_weight 1.0 \
    -dice_weight 0.1 \
    -img_folder /public/home/jd_shb/fromgithub/multimodal/OBELISK_V02/preprocess/datasets/LPBA40/train \
    -label_folder /public/home/jd_shb/fromgithub/multimodal/OBELISK_V02/preprocess/datasets/LPBA40/label \
    -train_scannumbers "11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40" \
    -val_scannumbers "1 2 3" \
    -img_name S?.delineation.skullstripped.nii.gz \
    -label_name S?.delineation.structure.label.nii.gz
```

### 预测

> **register_vm_3d.py**

运行：

```shell
python register_vm_3d.py \
    -output output/lpba_test/ \
    -model output/lpba/lpba40_best71.pth \
    -img_folder /public/home/jd_shb/fromgithub/multimodal/OBELISK_V02/preprocess/datasets/LPBA40/train \
    -label_folder /public/home/jd_shb/fromgithub/multimodal/OBELISK_V02/preprocess/datasets/LPBA40/label \
    -fix_number "5" \
    -mov_numbers "4 6 7 8 9 10" \
    -img_name S?.delineation.skullstripped.nii.gz \
    -label_name S?.delineation.structure.label.nii.gz
```

会保存 moved image、moved label、flow field，评估指标保存在 test.log。


## ANTs

### 运行

> **ants_baseline.py**
 
```shell
#!/bin/bash
#SBATCH -J ants_lpba
#SBATCH -p gpu1
#SBATCH -N 1
#SBATCH -n 20
#SBATCH -o logs/trainOutLPBAAnts.txt
#SBATCH -e logs/trainErrLPBAAnts.txt

cd $SLURM_SUBMIT_DIR

python ants_baseline.py \
    -dataset lpba \
    -img_folder ./dataset/LPBA40/train \
    -label_folder ./dataset/LPBA40/label \
    -mov_numbers "1 8 9 11 14 18 33" \
    -fix_number 7 \
    -img_name S?.delineation.skullstripped.nii.gz \
    -label_name S?.delineation.structure.label.nii.gz \
    -trans_typ SyN \
    -output output/ANTs_SyN/lpba/fix_7
```
