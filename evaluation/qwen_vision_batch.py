# -*- coding: utf-8 -*-
import os
import json
import re
import torch
import argparse
import wandb
from tqdm import tqdm
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

DEFAULT_CACHE_DIR = None
DEFAULT_CHECKPOINT_PATH = "Qwen/Qwen3-VL-8B-Instruct"

EVALUATION_TEMPLATE = """
You are an objective image evaluator. Your goal is to verify if the image content matches the provided text description.

Target Description: "{description}"

Please think step by step:
1. Analyze the image content carefully.
2. Compare the visual elements with the "Target Description".
3. Determine if the image strictly meets the requirements.

Finally, output your judgment in the following format:
If it matches, output <answer>Yes</answer>.
If it does not match, output <answer>No</answer>.
"""

def check_is_moe(model_name):
    pattern = r"A\d+B"
    match = re.search(pattern, model_name, re.IGNORECASE)
    if match:
        print(f"[Info] Detected MoE pattern '{match.group()}' in model name.")
        return True
    return False

def parse_args():
    parser = argparse.ArgumentParser(description="Qwen-VL Image Evaluation")
    parser.add_argument("--batch_config", type=str, default=None,
                        help="Path to a JSON file containing a list of vision evaluation tasks.")

    parser.add_argument("--image_folder", type=str, default="./figures/")
    parser.add_argument("--input_jsonl", type=str, default="./data/prompts.jsonl")
    parser.add_argument("--output_jsonl", type=str, default="./data/results.jsonl")
    
    parser.add_argument("--checkpoint_path", type=str, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--cache_dir", type=str, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--wandb_project", type=str, default="eval_vision")
    
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

def prepare_inputs_for_vllm(messages, processor):
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages,
        image_patch_size=processor.image_processor.patch_size,
        return_video_kwargs=True,
        return_video_metadata=True
    )
    mm_data = {}
    if image_inputs is not None: mm_data['image'] = image_inputs
    if video_inputs is not None: mm_data['video'] = video_inputs
    return {'prompt': text, 'multi_modal_data': mm_data, 'mm_processor_kwargs': video_kwargs}

def log_to_wandb(wandb_project, results_list, image_folder, task_config):
    if not results_list: return

    folder_name = image_folder.strip(os.sep).replace(os.sep, "_")
    run_name = f"eval_{folder_name}"

    print(f"\n[WandB] Initializing run: {run_name}")
    if wandb.run is not None:
        wandb.finish()
    wandb.init(project=wandb_project, name=run_name, config=task_config)

    table_columns = ["id", "category", "evaluation_result", "image", "prompt", "evaluation_response", "expect_en", "expect_zh"]
    table_data = []

    print("[WandB] Preparing table data...")
    for item in tqdm(results_list, desc="Uploading"):
        img_id = item.get('id')
        image_path = os.path.join(image_folder, f"{img_id}.png")
        wandb_img = wandb.Image(image_path, caption=f"ID: {img_id}") if os.path.exists(image_path) else None
        
        row = [
            item.get('id'), item.get('category', 'Unknown'), item.get('evaluation_result'),
            wandb_img, item.get('prompt'), item.get('evaluation_response'),
            item.get('expect_en'), item.get('expect_zh')
        ]
        table_data.append(row)

    wandb.log({"evaluation_results": wandb.Table(columns=table_columns, data=table_data)})
    
    valid_items = [x for x in results_list if x.get('evaluation_result') is not None]
    if valid_items:
        avg_score = sum([x['evaluation_result'] for x in valid_items]) / len(valid_items)
        wandb.log({"average_accuracy": avg_score})
        print(f"[WandB] Total Accuracy: {avg_score:.2%}")

        category_stats = {}
        for item in valid_items:
            cat = item.get('category', 'Unknown')
            if cat not in category_stats: category_stats[cat] = []
            category_stats[cat].append(item['evaluation_result'])
        
        category_metrics = {}
        for cat, scores in category_stats.items():
            category_metrics[f"accuracy_category_{cat}"] = sum(scores) / len(scores)
        wandb.log(category_metrics)

    wandb.finish()

def process_single_task(llm, processor, task_config, wandb_project):
    image_folder = task_config['image_folder']
    input_jsonl = task_config['input_jsonl']
    output_jsonl = task_config['output_jsonl']

    print(f"\n>>> Processing Task: {image_folder}")
    if not os.path.exists(input_jsonl):
        print(f"Skipping: {input_jsonl} not found.")
        return

    os.makedirs(os.path.dirname(output_jsonl), exist_ok=True)
    
    data_list = []
    vllm_inputs_list = []
    
    with open(input_jsonl, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in tqdm(lines, desc="Preparing Inputs"):
        if not line.strip(): continue
        item = json.loads(line)
        img_id = item.get('id')
        expect_en = item.get('expect_en', '')
        
        image_path = os.path.join(image_folder, f"{img_id}.png")
        if not os.path.exists(image_path): 
            image_path = os.path.join(image_folder, f"{img_id}.jpg")
        if not os.path.exists(image_path): continue
            
        eval_prompt = EVALUATION_TEMPLATE.format(description=expect_en)
        image_uri = f"file://{os.path.abspath(image_path)}"
        messages = [{"role": "user", "content": [{"type": "image", "image": image_uri}, {"type": "text", "text": eval_prompt}]}]

        try:
            input_data = prepare_inputs_for_vllm(messages, processor)
            vllm_inputs_list.append(input_data)
            data_list.append(item)
        except Exception as e:
            print(f"Error ID {img_id}: {e}")

    if not vllm_inputs_list:
        print("No valid inputs for this task.")
        return

    outputs = llm.generate(vllm_inputs_list, sampling_params=SamplingParams(temperature=0.0, max_tokens=512))

    for item, output in zip(data_list, outputs):
        text = output.outputs[0].text
        item['evaluation_response'] = text
        item['evaluation_result'] = extract_answer(text)

    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for item in data_list:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    log_to_wandb(wandb_project, data_list, image_folder, task_config)

def main():
    args = parse_args()
    
    tasks = []
    if args.batch_config:
        print(f"Loading batch config from {args.batch_config}...")
        with open(args.batch_config, 'r') as f:
            tasks = json.load(f)
    else:
        tasks = [{
            "image_folder": args.image_folder,
            "input_jsonl": args.input_jsonl,
            "output_jsonl": args.output_jsonl
        }]
    
    if not tasks: return

    print(f"Loading processor from {args.checkpoint_path}...")
    processor = AutoProcessor.from_pretrained(args.checkpoint_path, trust_remote_code=True)
    
    print(f"Initializing vLLM...")
    llm = LLM(
        model=args.checkpoint_path,
        mm_encoder_tp_mode="data",
        tensor_parallel_size=torch.cuda.device_count(),
        seed=42,
        download_dir=args.cache_dir,
        trust_remote_code=True,
        max_model_len=4096,
        limit_mm_per_prompt={"image": 1},
        enable_expert_parallel=check_is_moe(args.checkpoint_path),
    )

    for i, task in enumerate(tasks):
        print(f"--- [Batch Progress] Task {i+1}/{len(tasks)} ---")
        try:
            process_single_task(llm, processor, task, args.wandb_project)
        except Exception as e:
            print(f"[Error] Task failed: {e}")
            import traceback
            traceback.print_exc()

if __name__ == '__main__':
    main()