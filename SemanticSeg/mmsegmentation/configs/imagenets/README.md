# Semi-supervised Semantic Segmentation on the ImageNet-S dataset

[Large-scale Unsupervised Semantic Segmentation](https://arxiv.org/abs/2106.03149)

## Introduction

<!-- [ALGORITHM] -->

<a href="https://github.com/LUSSeg/ImageNetSegModel">Official Repo</a>

<a href="blob/main/mmseg/datasets/imagenets.py#L92">Code Snippet</a>

## Abstract

<!-- [ABSTRACT] -->

Based on the ImageNet dataset, the ImageNet-S dataset has 1.2 million training images and 50k high-quality semantic segmentation annotations to
support unsupervised/semi-supervised semantic segmentation on the ImageNet dataset. ImageNet-S dataset is available on [ImageNet-S](https://github.com/LUSSeg/ImageNet-S). More details about the dataset please refer to the [project page](https://LUSSeg.github.io/) or [paper link](https://arxiv.org/abs/2106.03149).

## Citation

```bibtex
@article{gao2022luss,
  title={Large-scale Unsupervised Semantic Segmentation},
  author={Gao, Shanghua and Li, Zhong-Yu and Yang, Ming-Hsuan and Cheng, Ming-Ming and Han, Junwei and Torr, Philip},
  journal=TPAMI,
  year={2022}
}
```

## Usage

To finetune with different pre-trained models, please convert keys following [`vit`](../vit/README.md).

## Results and models

### ImageNet-S

| Method | Backbone | pre-training epochs | pre-training mode | Crop Size | Lr schd | Mem (GB) | Inf time (fps) | mIoU | pre-trained                                                                                                         | config                                                                                                                                           | download                 |
| ------ | -------- | ------------------- | ----------------- | --------- | ------: | -------- | -------------- | ---: | ------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------ |
| MAE    | ViT-B/16 | 1600                | SSL               | 224x224   |    3600 |          |                |  41.3 | [pre-trained](https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth)                                | [config](https://github.com/LUSSeg/mmsegmentation/blob/master/configs/imagenets/fcn_mae-base_pretrained_fp16_8x32_224x224_36k_imagenets919.py)   | [model](<>) \| [log](<>) |
| MAE    | ViT-B/16 | 1600                | SSL+Sup           | 224x224   |    3600 |          |                |  60.9 | [pre-trained](https://dl.fbaipublicfiles.com/mae/finetune/mae_finetuned_vit_base.pth)                               | [config](https://github.com/LUSSeg/mmsegmentation/blob/master/configs/imagenets/fcn_mae-base_finetuned_fp16_8x32_224x224_36k_imagenets919.py)    | [model](<>) \| [log](<>) |
| SERE   | ViT-S/16 | 100                 | SSL               | 224x224   |    3600 |          |                |  59.9 | [pre-trained](https://github.com/LUSSeg/ImageNetSegModel/releases/download/vit/sere_pretrained_vit_small_ep100.pth) | [config](https://github.com/LUSSeg/mmsegmentation/blob/master/configs/imagenets/fcn_sere-small_pretrained_fp16_8x32_224x224_36k_imagenets919.py) | [model](<>) \| [log](<>) |
| SERE   | ViT-S/16 | 100                 | SSL+Sup           | 224x224   |    3600 |          |                |  41.4 | [pre-trained](https://github.com/LUSSeg/ImageNetSegModel/releases/download/vit/sere_finetuned_vit_small_ep100.pth)  | [config](https://github.com/LUSSeg/mmsegmentation/blob/master/configs/imagenets/fcn_sere-small_finetuned_fp16_8x32_224x224_36k_imagenets919.py)  | [model](<>) \| [log](<>) |
