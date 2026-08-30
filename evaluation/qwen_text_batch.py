# -*- coding: utf-8 -*-
import os
import json
import re
import torch
import argparse
import wandb
from tqdm import tqdm
from vllm import LLM, SamplingParams

# Required environment variable settings
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

# ================= Configuration Section =================
DEFAULT_CHECKPOINT_PATH = "Qwen/Qwen3-8B" 
DEFAULT_CACHE_DIR = None

EVALUATION_TEMPLATE = """
You are an objective logic evaluator. 
I will provide you with a "User Prompt" and a model's "Thought Process".
Your task is to judge whether the model's thought process implies or covers the meaning of a specific "Target Criteria".

=== User Prompt ===
{user_prompt}

=== Model Thought Process ===
{model_thought}

=== Target Criteria ===
{criteria}

=== Instruction ===
Please think step by step:
1. Analyze the "Target Criteria" to understand the core requirement.
2. Read the "Thought Process" carefully.
3. Determine if the thought process explicitly or implicitly covers the "Target Criteria". 
   - Note: It doesn't need to match word-for-word, but the logic/meaning must be present.

Finally, output your judgment in the following format:
If the criteria is met/implied, output <answer>Yes</answer>.
If the criteria is missed or contradicted, output <answer>No</answer>.
"""

def check_is_moe(model_name):
    pattern = r"A\d+B"
    match = re.search(pattern, model_name, re.IGNORECASE)
    if match:
        print(f"[Info] Detected MoE pattern '{match.group()}' in model name.")
        return True
    return False

def parse_args():
    parser = argparse.ArgumentParser(description="Thinking Process Evaluation")
    
    # New: Batch Config path
    parser.add_argument("--batch_config", type=str, default=None,
                        help="Path to a JSON file containing a list of evaluation tasks.")

    parser.add_argument("--prompts_file", type=str, default="./data/prompts.jsonl")
    parser.add_argument("--thoughts_file", type=str, default=None)
    parser.add_argument("--output_file", type=str, default="./data/thought_eval_results.jsonl")
    
    parser.add_argument("--checkpoint_path", type=str, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--cache_dir", type=str, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--wandb_project", type=str, default="eval_text")
    
    return parser.parse_args()

def extract_answer(response_text):
    match = re.search(r"<answer>(.*?)</answer>", response_text, re.IGNORECASE)
    if match:
        content = match.group(1).strip().lower()
        if "yes" in content: return 1
        elif "no" in content: return 0
    lower_text = response_text.lower()
    if "yes" in lower_text: return 1
    elif "no" in lower_text: return 0
    return 0 

def load_jsonl(file_path):
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
    except FileNotFoundError:
        print(f"[Error] File not found: {file_path}")
        return []
    return data

def merge_data_full(prompts_data, thoughts_data):
    prompts_map = {str(item['id']): item for item in prompts_data}
    merged_list = []
    for t_item in thoughts_data:
        tid = str(t_item.get('id'))
        if tid in prompts_map:
            p_item = prompts_map[tid]
            merged_item = p_item.copy()
            merged_item.update(t_item)
            merged_list.append(merged_item)
    return merged_list

def log_to_wandb_full(wandb_project, results_list, run_name, config_dict):
    if not results_list: return
    print(f"\n[WandB] Initializing run: {run_name}")
    
    # Reinitialize wandb
    if wandb.run is not None:
        wandb.finish()
        
    wandb.init(project=wandb_project, name=run_name, config=config_dict)

    all_keys = set().union(*(d.keys() for d in results_list))
    priority_keys = ['id', 'category', 'evaluation_result', 'evaluation_response', 'expect_en', 'thought', 'prompt']
    other_keys = sorted([k for k in all_keys if k not in priority_keys])
    table_columns = priority_keys + other_keys
    
    table_data = []
    for item in results_list:
        row = [item.get(col, None) for col in table_columns]
        table_data.append(row)

    eval_table = wandb.Table(columns=table_columns, data=table_data)
    wandb.log({"eval_results_full": eval_table})
    
    valid_items = [x for x in results_list if x.get('evaluation_result') is not None]
    if valid_items:
        total_scores = [x['evaluation_result'] for x in valid_items]
        avg_score = sum(total_scores) / len(total_scores)
        wandb.log({"average_accuracy": avg_score})
        print(f"[WandB] Total Accuracy: {avg_score:.2%}")

        category_stats = {}
        for item in valid_items:
            cat = item.get('category', 'Unknown')
            if cat not in category_stats: category_stats[cat] = []
            category_stats[cat].append(item['evaluation_result'])
        
        category_metrics = {}
        for cat, scores in category_stats.items():
            cat_avg = sum(scores) / len(scores)
            category_metrics[f"accuracy_category_{cat}"] = cat_avg
        wandb.log(category_metrics)

    wandb.finish()

def process_single_task(llm, tokenizer, task_config, wandb_project):
    """Process core logic for a single task"""
    prompts_file = task_config['prompts_file']
    thoughts_file = task_config['thoughts_file']
    output_file = task_config['output_file']

    print(f"\n>>> Processing Task: {thoughts_file}")
    
    prompts_data = load_jsonl(prompts_file)
    thoughts_data = load_jsonl(thoughts_file)
    data_list = merge_data_full(prompts_data, thoughts_data)
    
    if not data_list:
        print("Skipping task due to empty data.")
        return

    # Prepare Prompts
    prompts_list = []
    for item in data_list:
        input_text = EVALUATION_TEMPLATE.format(
            user_prompt=item.get('prompt', ''),
            model_thought=item.get('thought', ''),
            criteria=item.get('expect_en', '')
        )
        messages = [{"role": "user", "content": input_text}]
        prompt_str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts_list.append(prompt_str)

    # Inference
    sampling_params = SamplingParams(temperature=0.0, max_tokens=512)
    outputs = llm.generate(prompts=prompts_list, sampling_params=sampling_params)

    # Save results
    for item, output in zip(data_list, outputs):
        resp_text = output.outputs[0].text
        item['evaluation_response'] = resp_text
        item['evaluation_result'] = extract_answer(resp_text)
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for item in data_list:
            f_out.write(json.dumps(item, ensure_ascii=False) + '\n')

    # WandB
    # Auto-generate run name
    dir_path = os.path.dirname(thoughts_file)
    norm_path = os.path.normpath(dir_path).strip(os.sep).replace(os.sep, "_")
    if not norm_path: norm_path = "root"
    run_name = f"eval_{norm_path}"
    
    log_to_wandb_full(wandb_project, data_list, run_name, task_config)

def main():
    args = parse_args()
    
    # 1. Determine task list
    tasks = []
    if args.batch_config:
        print(f"Loading batch config from {args.batch_config}...")
        with open(args.batch_config, 'r') as f:
            tasks = json.load(f)
    else:
        # Single task mode
        tasks = [{
            "prompts_file": args.prompts_file,
            "thoughts_file": args.thoughts_file,
            "output_file": args.output_file
        }]

    if not tasks:
        print("No tasks to run.")
        return

    # 2. Initialize model (only once)
    print(f"Initializing vLLM with {torch.cuda.device_count()} GPUs...")
    llm = LLM(
        model=args.checkpoint_path,
        tensor_parallel_size=torch.cuda.device_count(),
        seed=42,
        download_dir=args.cache_dir,
        trust_remote_code=True,
        max_model_len=8192, 
        enable_expert_parallel=check_is_moe(args.checkpoint_path),
    )
    tokenizer = llm.get_tokenizer()

    # 3. Loop through tasks
    for i, task in enumerate(tasks):
        print(f"--- [Batch Progress] Task {i+1}/{len(tasks)} ---")
        try:
            process_single_task(llm, tokenizer, task, args.wandb_project)
        except Exception as e:
            print(f"[Error] Failed to process task {task}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == '__main__':
    main()