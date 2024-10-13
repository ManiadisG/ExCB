# ExCB: Efficient Unsupervised Visual Representation Learning with Explicit Cluster Balancing [ECCV 2024]

This is the official PyTorch implementation for [Efficient Unsupervised Visual Representation Learning with Explicit Cluster Balancing](https://arxiv.org/abs/2407.11168).

The repository will be updated in the coming days with the kNN and semi-supervised evaluation scripts.

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
| ResNet50 | 1024 | 400 | :white_check_mark: | 71.6% | 76.6% | [Link](https://drive.google.com/file/d/1K_kXKKDngtNxLjbzerkUcip9ZX04MLoQ/view?usp=sharing) | `resnet50_ep400.yaml` |
| ViT-S/16 | 1024 | 800 | :white_check_mark: | [x] | 77.2% | [Link](https://drive.google.com/file/d/17QUPjlspI-RN1JiGhI684jmpm5-6nZVv/view?usp=sharing) | `vit_s16_ep800.yaml` |

## How to run

Experiments are run using bash scripts. Each type of experiment (pretraining, linear probing, etc.) reads from the corresponding yml config file `main_config.yml`.
Via the "Preset" argument in each bash script can select customized yml configs for the experiment they want to run, that overwrite the arguments of `main_config.yml`.

For example, to run ExCB for pretraining for ResNet50 and 400 epochs on 4 GPUs, run the following command, where `resnet50_ep400` is the overriding config file and `R50_pretraining` is the name of the experiment:

```
bash run.sh 4 resnet50_ep400 R50_pretraining
```

Arguments can be further customized via the "Complex_arg" input to the bash script, separated via double lower dash symbols. Additionally, one can specify the exact GPUs they want the experiment to run on. E.g. to run the above pretraining for 100 epochs and batch size 512 on GPUs 1 & 2, run the following:

```
bash run.sh 1,2 resnet50_ep400 R50_pretraining_ep100_bs512 epochs__100__batch_size__512
```

To run linear evaluation with a given checkpoint, run the following command:

```
bash run_linear_eval.sh 4 linear_eval imagenet_resnet <path_to_checkpoint>
```

## Requirements

The required packages to run this code can be found in `requirements.txt`.

Experiments were run using `pytorch==2.1.2` and `torchvision==0.16.2`.

## Acknowledgement
Our implementation uses code from [DINO](https://github.com/facebookresearch/dino) and [MIRA](https://github.com/movinghoon/MIRA). 

## Citation
If you find this repository useful, please cite:
```
@inproceedings{maniadis2024excb,
    title = {Efficient Unsupervised Visual Representation Learning with Explicit Cluster Balancing},
    author = {Maniadis Metaxas, Ioannis and Tzimiropoulos, Georgios and Patras, Ioannis},
    booktitle = {European Conference on Computer Vision (ECCV)},
    year={2024}
}
```