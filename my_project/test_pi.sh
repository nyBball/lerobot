export HF_HUB_OFFLINE=1
export LIBERO_CONFIG_PATH=/home/work/my_config/libero/
export TOKENIZERS_PARALLELISM=false

lerobot-eval\
    --policy.path=/home/work/lerobot/pi05/my_ft/train/dev3/1/checkpoints/009000/pretrained_model\
    --output_dir=/home/work/save/lerobot/pi05/my_ft/eval/dev3/1/9k_step/action_20_obs_4_interval_5_trueaction\
    --env.type=libero\
    --env.task=libero_spatial,libero_object,libero_goal,libero_10\
    --eval.batch_size=1\
    --eval.n_episodes=10\
    --job_name=pi05_libero_eval\
    --policy.device=cuda\
    --policy.n_action_steps=20\
    --policy.n_obs_steps=2\
    --policy.obs_frame_interval=1\
    --policy.tokenizer_max_length=300\
    --policy.num_inference_steps=10\
    --policy.compile_model=false\
    --env.max_parallel_tasks=1