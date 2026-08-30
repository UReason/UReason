python evaluation/qwen_text_batch.py \
    --batch_config eval_cfgs/text_config.json \
    --checkpoint_path ${checkpoint_path} \
    --cache_dir ${cache_dir} \
    --wandb_project eval_text