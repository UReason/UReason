########## 1. Direct Generation ##########
python Bagel/gen_batch_dist.py \
--prompt_file ./data/direct_${dataset}.jsonl \
--output_dir ./results/${model}/direct/direct_${dataset}/ \
--model_path ${model_path} \
--use_wandb \
--wandb_project ${model} \
--wandb_run_name direct_${dataset} \


########## 2. Reasoning-Guided Generation ##########
python Bagel/gen_batch_dist.py \
--prompt_file data/${dataset}.jsonl \
--output_dir ./results/${model}/think/${dataset} \
--model_path ${model_path} \
--use_wandb \
--wandb_project ${model} \
--wandb_run_name think_${dataset} \
--enable_think


########## 3. De-contextualized Generation ##########
python ${model}/extract_thought.py results/${model}/think/${dataset}/${dataset}.jsonl
mkdir -p ./results/${model}/think-generated_prompts/${dataset}/
cp results/${model}/think/${dataset}/think-generated_prompts.jsonl ./results/${model}/think-generated_prompts/${dataset}/prompts.jsonl

python Bagel/gen_batch_dist.py \
--prompt_file ./results/${model}/think-generated_prompts/${dataset}/prompts.jsonl \
--output_dir ./results/${model}/think-generated_prompts/${dataset}/ \
--model_path ${model_path} \
--use_wandb \
--wandb_project ${model} \
--wandb_run_name think-generated_prompts_${dataset}