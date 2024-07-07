# ExCB: Efficient Unsupervised Visual Representation Learning with Explicit Cluster Balancing [ECCV 2024]

This is the official PyTorch implementation for [Efficient Unsupervised Visual Representation Learning with Explicit Cluster Balancing](https://arxiv.org/abs/2407.11168).

Code and models will be uploaded soon!

[![arXiv](https://img.shields.io/badge/arXiv-2407.11168-red)](https://arxiv.org/abs/2407.11168) 

## Abstract

Self-supervised learning has recently emerged as the preeminent pretraining paradigm across and between modalities, with remarkable results. 
In the image domain specifically, group (or cluster) discrimination has been one of the most successful methods.
However, such frameworks need to guard against heavily imbalanced cluster assignments to prevent collapse to trivial solutions.
Existing works typically solve this by reweighing cluster assignments to promote balance, or with offline operations (e.g. regular re-clustering) that prevent collapse.
However, the former typically requires large batch sizes, which leads to increased resource requirements, and the latter introduces scalability issues with regard to large datasets.
In this work, we propose **ExCB**, a framework that tackles this problem with a novel cluster balancing method.
**ExCB** estimates the relative size of the clusters across batches and balances them by adjusting cluster assignments, proportionately to their relative size and in an online manner.
Thereby, it overcomes previous methods' dependence on large batch sizes and is fully online, and therefore scalable to any dataset.
We conduct extensive experiments to evaluate our approach and demonstrate that **ExCB**: **a)** achieves state-of-the-art results with significantly reduced resource requirements compared to previous works, **b)** is fully online, and therefore scalable to large datasets, and **c)** is stable and effective even with very small batch sizes.

![ExCB overview](/assets/main.png "ExCB overview")

## Pretrained models

| Architecture | Batch Size | Epochs | Multi-Crop | k-NN Acc. | Linear Acc. | Model | Train Config |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| ResNet50 | 1024 | 400 | :white_check_mark: | 71.6% | 76.5% | [x] | [x] |
| ViT-S/16 | 1024 | 800 | :white_check_mark: | [x] | 77.1% | [x] | [x] |


## Acknowledgement
Our implementation uses code from [DINO](https://github.com/facebookresearch/dino) and [MIRA](https://github.com/movinghoon/MIRA). 

## Citation
If you find this repository useful, please cite:
```
@inproceedings{maniadis2024efficient,
    title = {Efficient Unsupervised Visual Representation Learning with Explicit Cluster Balancing},
    author = {Maniadis Metaxas, Ioannis and Tzimiropoulos, Georgios and Patras, Ioannis},
    booktitle = {European Conference on Computer Vision (ECCV)},
    year={2024}
}
```