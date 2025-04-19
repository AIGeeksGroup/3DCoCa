# 3DCoCa
This is the code repository for the paper:
> **3D CoCa: Contrastive Learners are 3D Captioners**
>
> [Ting Huang](https://github.com/Believeht029)\*, [Zeyu Zhang](https://steve-zeyu-zhang.github.io/)\*†, [Yemin Wang](https://github.com/Clare-1)\* and [Hao Tang](https://scholar.google.com/citations?user=9zJkeEMAAAAJ&hl=en)\**
>
> \*Equal contribution. †Project lead. \**Corresponding author
>
>
> **[[arXiv]](https://arxiv.org/abs/2504.09518)** **[[Paper with Code]](https://paperswithcode.com/paper/3d-coca-contrastive-learners-are-3d)** **[[HF Paper]](https://huggingface.co/papers/2504.09518)**

<center class='img'>
<img title="Conceptual homepage figure for 3D CoCa, highlighting its architecture (left) and performance (right). Left: The 3D CoCa model unifies contrastive learning and multimodal captioning in one framework. Right:Radar chart comparison of 3D CoCa and previous methods Scan2Cap~\cite{scan2cap_2021}, 3DJCG~\cite{3djcg2022}, 3D-VLP~\cite{3dvlp2024}, Vote2Cap-DETR~\cite{vote2cap2023}, Vote2Cap-DETR++~\cite{vote2cap++2024} on the ScanRefer~\cite{chen2020scanrefer} benchmark." src="https://github.com/AIGeeksGroup/3DCoCa/blob/main/image.png" width="100%">
</center>

## Citation

If you use any content of this repo for your work, please cite the following our paper:
```
@article{huang20253d,
  title={3D CoCa: Contrastive Learners are 3D Captioners},
  author={Huang, Ting and Zhang, Zeyu and Wang, Yemin and Tang, Hao},
  journal={arXiv preprint arXiv:2504.09518},
  year={2025}
}
```

## Introduction
3D captioning, which aims to describe the content of 3D scenes in natural language, remains highly challenging due to the inherent sparsity of point clouds and weak cross-modal alignment in existing methods. To address these challenges, we propose 3D CoCa, a novel unified framework that seamlessly combines contrastive vision-language learning with 3D caption generation in a single architecture. Our approach leverages a frozen CLIP vision-language backbone to provide rich semantic priors, a spatially-aware 3D scene encoder to capture geometric context, and a multi-modal decoder to generate descriptive captions. Unlike prior two-stage methods that rely on explicit object proposals, 3D CoCa jointly optimizes contrastive and captioning objectives in a shared feature space, eliminating the need for external detectors or handcrafted proposals. This joint training paradigm yields stronger spatial reasoning and richer semantic grounding by aligning 3D and textual representations. Extensive experiments on the ScanRefer and Nr3D benchmarks demonstrate that 3D CoCa significantly outperforms current state-of-the-arts by 10.2\% and 5.76\% in CIDEr&#8203;@0.5IoU, respectively.

<!-- ![image](https://github.com/AIGeeksGroup/3DCoCa/blob/main/image1.png)-->

## Environment Setup
You can set up your own conda virtual environment by running the commands below.

```bash
# create a clean conda environment from scratch
conda create --name 3DCoCa python=3.8
conda activate 3DCoCa

# install pip
conda install ipython
conda install pip

# install required packages
pip install -r requirements.txt
```

## Training
### Import Dataset Path
There are two datasets that need to set paths. The Scanrefer dataset sets the `DATASET_ROOT_DIR` and `DATASET_METADATA_DIR` global variables in the `datasets/scene_scanrefer.py` file. The Nr3D dataset also sets two global variables in the `datasets/scene_nr3d.py` file.

Please modify the paths to match your actual dataset paths, set the training parameters, and then start model training.

### Start Training
```bash
# w/o 2D input
python main.py --use_color --use_normal --checkpoint_dir ckpt/3DCoCa
# w/ 2D input
python main.py --use_color --use_multiview --checkpoint_dir ckpt_2D/3DCoCa
```

## Evaluation

There are two datasets that need to set paths. The Scanrefer dataset sets the `DATASET_ROOT_DIR` and `DATASET_METADATA_DIR` global variables in the `datasets/scene_scanrefer.py` file. The Nr3D dataset also sets two global variables in the `datasets/scene_nr3d.py` file.

Please modify the paths to match your actual dataset paths, set the training parameters, and then start model training.

### Start testing
```bash
# w/o 2D input
python main.py --use_color --use_normal --test_ckpt ckpt/3DCoCa/checkpoint_best.pth --test_caption
# w/ 2D input
python main.py --use_color --use_multiview --test_ckpt ckpt_2D/3DCoCa/checkpoint_best.pth --test_caption
```
