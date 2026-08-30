import os
import sys

# ==============================================================================
# 🚀 Ultimate Fix: Physical Isolation Strategy (Must be placed before import torch)
# ==============================================================================
# Read LOCAL_RANK automatically passed by torchrun/accelerate
local_rank = os.environ.get("LOCAL_RANK")

if local_rank is not None:
    # Force current process to only see this one physical GPU
    # For example: Process 3 only sees physical GPU 3, but internally it sees "cuda:0"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(local_rank)
    print(f"🔒 Process restricted to Physical GPU: {local_rank}")
else:
    # If no LOCAL_RANK, it's not distributed running, default to all visible
    print("⚠️ No LOCAL_RANK found. Assuming single process or manually managed.")
# =======================
import argparse
import random
import numpy as np
import torch
import json
import glob
from tqdm import tqdm
from copy import deepcopy
from typing import Dict, Any

import wandb
from PIL import Image
from accelerate import Accelerator, init_empty_weights, load_checkpoint_and_dispatch

# Import internal project dependencies
from data.transforms import ImageTransform
from data.data_utils import add_special_tokens
from modeling.bagel import (
    BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM, SiglipVisionConfig, SiglipVisionModel
)
from modeling.qwen2 import Qwen2Tokenizer
from modeling.autoencoder import load_ae
from inferencer import InterleaveInferencer

# ================= Configuration Section =================
DEFAULT_MODEL_PATH = None 
SEED = 42

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_model_and_components(model_path, accelerator):
    if accelerator.is_main_process:
        print(f"Loading model from {model_path}...")

    # Print current device status, you should see device_count is 1
    # and accelerator.device should be cuda:0
    print(f"[Rank {accelerator.process_index}] Visible Devices: {torch.cuda.device_count()} | My Device: {accelerator.device}")

    llm_config = Qwen2Config.from_json_file(os.path.join(model_path, "llm_config.json"))
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module = "Qwen2MoTDecoderLayer"

    vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_path, "vit_config.json"))
    vit_config.rope = False
    vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1

    # VAE directly placed on cuda:0 (because each process only owns one card)
    vae_model, vae_config = load_ae(local_path=os.path.join(model_path, "ae.safetensors"))
    vae_model = vae_model.to("cuda:0").to(torch.bfloat16).eval()

    config = BagelConfig(
        visual_gen=True,
        visual_und=True,
        llm_config=llm_config, 
        vit_config=vit_config,
        vae_config=vae_config,
        vit_max_num_patch_per_side=70,
        connector_act='gelu_pytorch_tanh',
        latent_patch_size=2,
        max_latent_size=64,
    )

    with init_empty_weights():
        language_model = Qwen2ForCausalLM(llm_config)
        vit_model = SiglipVisionModel(vit_config)
        model = Bagel(language_model, vit_model, config)
        model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)

    tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

    vae_transform = ImageTransform(1024, 512, 16)
    vit_transform = ImageTransform(980, 224, 14)

    # ================= Key Modification =================
    # Regardless of which Rank I am, I load the model to "cuda:0"
    # Because the OS level has already mapped "cuda:0" to the correct physical card
    device_map = {"": "cuda:0"}
    # ==========================================

    # ================= Core Fix: Simple and Direct Logic =================
    if "BAGEL-7B-MoT" in model_path:
        # Old model uses ema
        checkpoint_file = "ema.safetensors"
    elif "Bagel-Zebra-CoT" in model_path:
        # New model must use bf16
        checkpoint_file = "model_bf16.safetensors"
    else:
        # Default for other cases
        checkpoint_file = "model.safetensors"
    
    checkpoint_path = os.path.join(model_path, checkpoint_file)
    
    if accelerator.is_main_process:
        print(f"👉 Target Checkpoint: {checkpoint_file}")
    # ==============================================================


    model = load_checkpoint_and_dispatch(
        model,
        checkpoint=checkpoint_path,
        device_map=device_map,
        offload_buffers=False,
        dtype=torch.bfloat16,
        force_hooks=True,
    )
    model = model.eval()
    
    if accelerator.is_main_process:
        print("Model loaded successfully.")

    return model, vae_model, tokenizer, new_token_ids, vae_transform, vit_transform

def process_jsonl(file_path):
    """Generator to read jsonl file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: 
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                pass

def main():
    parser = argparse.ArgumentParser(description="BAGEL Batch Image Generation from JSONL (Distributed)")
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH, help="Path to model weights")
    parser.add_argument("--prompt_file", type=str, required=True, help="Path to .jsonl file containing prompts")
    parser.add_argument("--output_dir", type=str, default="batch_output", help="Directory to save generated images")
    parser.add_argument("--use_wandb", action="store_true", help="Enable WandB logging")
    parser.add_argument("--wandb_project", type=str, default="bagel-batch-generation", help="WandB project name")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="WandB run name (used as group name)")
    parser.add_argument("--enable_think", action="store_true", help="Enable thinking mode in inferencer")
    parser.add_argument("--debug", action="store_true", help="Enable debug output for WandB table")
    parser.add_argument("--injection_text", action="store_true", help="Text to inject before generation")
    args = parser.parse_args()

    # Physical isolation strategy (keep your code unchanged)
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank != -1:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(local_rank)
    
    accelerator = Accelerator()
    set_seed(SEED) 
    
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # Load model
    model, vae_model, tokenizer, new_token_ids, vae_transform, vit_transform = load_model_and_components(args.model_path, accelerator)

    inferencer = InterleaveInferencer(
        model=model, 
        vae_model=vae_model, 
        tokenizer=tokenizer, 
        vae_transform=vae_transform, 
        vit_transform=vit_transform, 
        new_token_ids=new_token_ids,
        model_path=args.model_path
    )

    inference_hyper = dict(
        max_think_token_n=1000,
        do_sample=False,
        cfg_text_scale=4.0,
        cfg_img_scale=1.0,
        cfg_interval=[0.4, 1.0],
        timestep_shift=3.0,
        num_timesteps=50,
        cfg_renorm_min=0.0,
        cfg_renorm_type="global",
    )

    # =================================================================
    # [Modification 1]: WandB Initialization - Remove is_main_process restriction, use Group
    # =================================================================
    if args.use_wandb:
        # Define Group ID: if user didn't pass run_name, give a default value to ensure all cards use the same group name
        group_id = args.wandb_run_name if args.wandb_run_name else "distributed_batch_run"
        
        # Define current process's Run Name
        rank_run_name = f"{group_id}_rank_{accelerator.process_index}"

        wandb.init(
            project=args.wandb_project,
            group=group_id,        # Key: all cards share the same group
            job_type="generation", # Mark task type
            name=rank_run_name,    # Key: each card has unique name
            config={
                "model_path": args.model_path,
                "prompt_file": args.prompt_file,
                "seed": SEED,
                "rank": accelerator.process_index, # Record rank
                "enable_think": args.enable_think,
                **inference_hyper
            },
            reinit=True
        )
        if accelerator.is_main_process:
            print(f"WandB initialized for Group: {group_id}")

    # Data reading section (keep unchanged)
    try:
        all_items = list(process_jsonl(args.prompt_file))
    except FileNotFoundError:
        return

    my_items = all_items[accelerator.process_index::accelerator.num_processes]
    
    local_results = []
    
    # Progress bar logic remains unchanged
    progress_bar = tqdm(my_items, desc=f"GPU {accelerator.process_index}", disable=not accelerator.is_local_main_process)
    
    for idx, item in enumerate(progress_bar):
        prompt_text = item.get("prompt")
        if not prompt_text:
            continue
            
        sample_id = item.get("id", str(item.get("id_in_file", idx))) 
        
        # Ensure sample_id doesn't conflict across multiple cards (although filenames may conflict, but this is just for logging)
        # Suggestion: If original data id is not unique, add rank suffix here
        # sample_id = f"{sample_id}_r{accelerator.process_index}" 
        
        if True:
            output_dict = inferencer(text=prompt_text, think=args.enable_think, injection_text="The refined prompt is:" if args.injection_text else None, **inference_hyper)
            thought_text = output_dict.get('text', "")
            img = output_dict.get('image')
            
            if isinstance(img, list):
                img = img[0] if len(img) > 0 else None

            if img is not None:
                if not isinstance(img, Image.Image):
                    continue
                
                # Filename save logic
                save_filename = f"{sample_id}.png"
                save_filename = "".join([c for c in save_filename if c.isalnum() or c in (' ', '.', '-', '_')]).strip()
                save_path = os.path.join(args.output_dir, save_filename)
                
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                img.save(save_path)
                
                # =================================================================
                # [Modification 2]: WandB Log - Remove is_main_process, all cards upload
                # =================================================================
                if args.use_wandb:
                    try:
                        short_prompt = prompt_text
                        wandb.log({
                            f"generated_images/{sample_id}": wandb.Image(
                                img,  # Directly use PIL Image object
                                caption=f"ID: {sample_id}\n{short_prompt}"
                            ),
                            "global_step": idx 
                        })
                    except Exception as e:
                        print(f"WandB logging failed: {e}")
                
                local_results.append({
                    "id": sample_id,
                    "prompt": prompt_text,
                    "thought": thought_text
                })
        
    # Save temporary results (keep unchanged)
    temp_filename = os.path.join(args.output_dir, f"temp_results_rank_{accelerator.process_index}.jsonl")
    with open(temp_filename, 'w', encoding='utf-8') as f:
        for res in local_results:
            f.write(json.dumps(res, ensure_ascii=False) + '\n')
    
    accelerator.wait_for_everyone()
    
    # =================================================================
    # [Modification 3]: WandB Table - Still recommend only main process uploads summary table
    # Reason: Table is a global view, if each Rank uploads a Table, it's wasteful and can't see all in one table
    # =================================================================
    if accelerator.is_main_process:
        input_filename = os.path.basename(args.prompt_file)
        final_output_path = os.path.join(args.output_dir, input_filename)
        
        print(f"\nMerging results from all GPUs to: {final_output_path}")
        
        all_results_for_table = []
        
        with open(final_output_path, 'w', encoding='utf-8') as f_out:
            temp_files = glob.glob(os.path.join(args.output_dir, "temp_results_rank_*.jsonl"))
            for t_file in temp_files:
                with open(t_file, 'r', encoding='utf-8') as f_in:
                    for line in f_in:
                        f_out.write(line)
                        if args.use_wandb:
                            try:
                                all_results_for_table.append(json.loads(line))
                            except: pass
                os.remove(t_file)
                
        print(f"✓ All results merged.")
        
        if args.use_wandb and len(all_results_for_table) > 0:
            print("Building WandB Summary Table...")
            # Note: Here we only log, but because the main process also initialized its own WandB run,
            # this Table will be uploaded to the main process's Run.
            # This is fine in the Group view.
            wandb_table = wandb.Table(columns=["ID", "Prompt", "Thought Process"])
            for res in all_results_for_table:
                wandb_table.add_data(str(res["id"]), res["prompt"], res["thought"])
            
            # Use commit=False to avoid immediately ending
            wandb.log({"generation_results_table": wandb_table})

    # =================================================================
    # [Modification 4]: Finish - All processes need to finish
    # =================================================================
    if args.use_wandb:
        wandb.finish()

    if accelerator.is_main_process:
        print(f"\n{'='*60}")
        print(f"Batch generation completed!")
        print(f"{'='*60}")

if __name__ == "__main__":
    main()