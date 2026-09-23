# Reasoning Diffusion for Unpaired Test Time Out-of-distribution Text-Image to Video Generation
> Zirui Pan, Xin Wang, Yipeng Zhang, Hong Chen, Kecheng Zheng, Wenwu Zhu
> 
> Department of Computer Science and Technology, Tsinghua University & BNRist & Ant Research

## Overview
This is the official implementation of CVPR 2026 paper [Reasoning Diffusion for Unpaired Test Time Out-of-distribution Text-Image to Video Generation](https://openaccess.thecvf.com/content/CVPR2026/papers/Pan_Reasoning_Diffusion_for_Unpaired_Test_Time_Out-of-distribution_Text-Image_to_Video_CVPR_2026_paper.pdf).

## Environment Setup
```bash
git clone git@github.com:pzrain/ReasonDiff.git
conda create -n reasondiff python=3.10
conda activate reasondiff
cd ReasonDiff
pip install -e .
```

## Training
1. Data preprocessing. A sample is provided in `metadata.csv`. You should create a similar `.csv` file under the root directory of the dataset.
2. Model preparation: Download pretrained Wan2.1-I2V-14B-480 weights to `models/Wan-AI/`.
3. Train:
```bash
bash examples/wanvideo/run.sh
```

## Inference
```bash
python wan_14b_image_to_video.py
```
Prepare your condition image and text prompts under `assets`.

## Acknowledgement
This codebase is built upon [Diffsynth-Studio](https://github.com/modelscope/diffsynth-studio). The method is built upon [Wan2.1](https://github.com/Wan-Video/Wan2.1).