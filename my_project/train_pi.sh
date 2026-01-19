#!/usr/bin/env bash

export HF_HUB_OFFLINE=1
export LIBERO_CONFIG_PATH=/home/work/my_config/libero/
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True # 减少内存碎片
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

accelerate launch --num_processes 8 --mixed_precision bf16 /home/work/src/lerobot/src/lerobot/scripts/lerobot_train.py\
    --policy.push_to_hub=false\
    --save_freq=3000\
    --dataset.repo_id=/home/work/data/libero\
    --policy.type=pi05\
    --output_dir=/home/work/save/lerobot/pi05/my_ft/train\
    --job_name=pi05_training\
    --policy.pretrained_path=/home/work/ckpt/pi05_base\
    --policy.compile_model=false\
    --policy.gradient_checkpointing=true\
    --policy.dtype=bfloat16\
    --policy.n_obs_steps=4\
    --policy.obs_frame_interval=5\
    --policy.queue_filling_aug_prob=0.3\
    --steps=30000\
    --policy.device=cuda\
    --batch_size=32\
    --policy.optimizer_lr=5e-5\
    --policy.scheduler_warmup_steps=10000\
    --policy.scheduler_decay_steps=30000\
    --policy.scheduler_decay_lr=5e-5\
    --wandb.enable=false\
    --eval_freq=0\
    --log_freq=10