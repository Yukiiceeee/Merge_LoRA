import os
import json
import argparse
import subprocess
from typing import Dict, List

def load_data_config(config_path: str) -> Dict:
    """Load the data configuration file."""
    with open(config_path, 'r') as f:
        return json.load(f)

def train_all_adapters(
    model_path: str,
    config_path: str,
    output_dir: str,
    peft_type: str = "lora",
    lora_r: int = 8,
    lora_alpha: float = 16,
    lora_dropout: float = 0.05,
    batch_size: int = 4,
    learning_rate: float = 5e-5,
    num_epochs: int = 3,
    max_length: int = 2048,
    use_deepspeed: bool = False,
    domains: List[str] = None,
    tasks: Dict[str, List[str]] = None
):
    """Train adapters for all tasks or selected tasks."""
    # Load data configuration
    data_config = load_data_config(config_path)
    
    # Filter domains if specified
    if domains:
        data_config = {k: v for k, v in data_config.items() if k in domains}
    
    # Count total tasks to train
    total_tasks = 0
    for domain, domain_tasks in data_config.items():
        if tasks and domain in tasks:
            domain_tasks = {k: v for k, v in domain_tasks.items() if k in tasks[domain]}
            data_config[domain] = domain_tasks
        total_tasks += len(domain_tasks)
    
    print(f"Starting training for {total_tasks} tasks across {len(data_config)} domains")
    
    # Train adapter for each task
    for domain, domain_tasks in data_config.items():
        for task, data_path in domain_tasks.items():
            print(f"\n=== Training adapter for {domain}/{task} ===")
            
            # Get train data path
            train_data_path = data_path
            if isinstance(data_path, dict):
                train_data_path = data_path.get("train", "")
            
            # Get eval data path if available
            eval_data_path = ""
            eval_args = ""
            if isinstance(data_path, dict) and "val" in data_path:
                eval_data_path = data_path["val"]
                eval_args = f"--eval_data_path {eval_data_path} --do_eval true"
            
            # Construct command
            command = f"""
            python train_adapter.py \
                --model_name_or_path {model_path} \
                --domain {domain} \
                --task {task} \
                --train_data_path {train_data_path} \
                {eval_args} \
                --output_dir {output_dir} \
                --peft_type {peft_type} \
                --lora_r {lora_r} \
                --lora_alpha {lora_alpha} \
                --lora_dropout {lora_dropout} \
                --per_device_train_batch_size {batch_size} \
                --per_device_eval_batch_size {batch_size} \
                --gradient_accumulation_steps 4 \
                --model_max_length {max_length} \
                --learning_rate {learning_rate} \
                --num_train_epochs {num_epochs} \
                --logging_steps 10 \
                --save_strategy "epoch" \
                --save_total_limit 2 \
                --use_deepspeed {str(use_deepspeed).lower()} \
                --do_train true
            """
            
            # Execute command
            print(f"Executing command: {command}")
            try:
                subprocess.run(command, shell=True, check=True)
                print(f"Successfully trained adapter for {domain}/{task}")
            except subprocess.CalledProcessError as e:
                print(f"Failed to train adapter for {domain}/{task}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train adapters for multiple tasks")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the base model")
    parser.add_argument("--config_path", type=str, required=True, help="Path to the data config file")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save adapters")
    parser.add_argument("--peft_type", type=str, default="lora", help="PEFT type (lora, dora, pissa, rslora)")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=float, default=16, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument("--batch_size", type=int, default=4, help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--max_length", type=int, default=2048, help="Maximum sequence length")
    parser.add_argument("--use_deepspeed", action="store_true", help="Use DeepSpeed")
    parser.add_argument("--domains", type=str, nargs="*", help="Specific domains to train (optional)")
    parser.add_argument("--tasks_config", type=str, help="Path to tasks filter config (optional)")
    
    args = parser.parse_args()
    
    # Load tasks filter if provided
    tasks = None
    if args.tasks_config:
        with open(args.tasks_config, 'r') as f:
            tasks = json.load(f)
    
    train_all_adapters(
        model_path=args.model_path,
        config_path=args.config_path,
        output_dir=args.output_dir,
        peft_type=args.peft_type,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        max_length=args.max_length,
        use_deepspeed=args.use_deepspeed,
        domains=args.domains,
        tasks=tasks
    )