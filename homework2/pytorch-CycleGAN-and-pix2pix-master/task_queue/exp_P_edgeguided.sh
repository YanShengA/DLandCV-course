#!/bin/bash
echo "启动 Exp P (Edge-Guided)"
# 核心参数: --use_edge_map
# 目的: 将真实照片的 Canny 边缘图作为额外输入通道，引导生成器关注物体轮廓
python train.py \
    --dataroot ./datasets/cityscapes \
    --name cityscapes_expP_edgeguided \
    --model pix2pix \
    --direction BtoA \
    --use_edge_map \
    --batch_size 32 \
    --gan_mode lsgan \
    --gpu_ids 0 \
    --n_epochs 100 --n_epochs_decay 100 \
    > logs/expP_edgeguided.log 2>&1
