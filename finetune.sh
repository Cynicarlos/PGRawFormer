#!/bin/bash

# 指定要使用的 Conda 环境
ENV_NAME="general"

# 指定要使用的 GPU 编号
GPU_ID=0

# 创建一个新的 tmux 会话
SESSION_NAME="general"
tmux new-session -d -s $SESSION_NAME

# 激活 Conda 环境
tmux send-keys -t $SESSION_NAME "source activate ${ENV_NAME}" C-m

# 设置环境变量，指定要使用的 GPU
tmux send-keys -t $SESSION_NAME "export CUDA_VISIBLE_DEVICES=${GPU_ID}" C-m

# 运行训练脚本
tmux send-keys -t $SESSION_NAME "python finetune.py -cfg ./configs/ELD_SonyA7S2.yaml > finetune_eld1.log 2>&1" C-m
tmux send-keys -t $SESSION_NAME "python finetune.py -cfg ./configs/ELD_NikonD850.yaml > finetune_eld2.log 2>&1" C-m
tmux send-keys -t $SESSION_NAME "python finetune.py -cfg ./configs/ELD_CanonEOS70D.yaml > finetune_eld3.log 2>&1" C-m
tmux send-keys -t $SESSION_NAME "python finetune.py -cfg ./configs/ELD_CanonEOS700D.yaml > finetune_eld4.log 2>&1" C-m

tmux send-keys -t $SESSION_NAME "python test_eld_finetuned.py --with_metainfo --merge_test > finetune_test.log 2>&1" C-m