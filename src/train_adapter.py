import copy
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List
import os
import json
import torch
import transformers
from torch.utils.data import Dataset
from transformers import Trainer
from peft import LoraConfig, get_peft_model
from peft.utils import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING as LORA_TARGET_MAP
from constants import (
    IGNORE_INDEX,
    DEFAULT_PAD_TOKEN,
    DEFAULT_EOS_TOKEN,
    DEFAULT_BOS_TOKEN,
    DEFAULT_UNK_TOKEN,
    PROMPT_DICT
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="/path/to/your/base/model")
    peft_type: Optional[str] = field(default="lora")
    lora_r: Optional[int] = field(default=8)
    lora_alpha: Optional[float] = field(default=16)
    lora_dropout: Optional[float] = field(default=0.05)


@dataclass
class DataArguments:
    domain: str = field(default=None, metadata={"help": "Domain name"})
    task: str = field(default=None, metadata={"help": "Task name"})
    train_data_path: str = field(default=None, metadata={"help": "Path to the training data"})
    eval_data_path: Optional[str] = field(default=None, metadata={"help": "Path to the evaluation data"})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default="./cache")
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=2048,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    use_deepspeed: bool = field(default=False)


def get_target(model_type, named_modules) -> List[str]:
    target_modules = LORA_TARGET_MAP.get(model_type, [])
    if not target_modules:
        cls = torch.nn.Linear
        lora_module_names = {name.split('.')[-1] for name, module in named_modules if isinstance(module, cls)}
        if "lm_head" in lora_module_names:
            lora_module_names.remove("lm_head")
        return list(lora_module_names)
    return target_modules


def load_model_and_tokenizer(model_args: ModelArguments, training_args: TrainingArguments) -> tuple:
    model_kwargs = {
        "cache_dir": training_args.cache_dir,
        "torch_dtype": 'auto',
        "trust_remote_code": True
    }
    if not training_args.use_deepspeed:
        model_kwargs["device_map"] = "auto"
    else:
        logger.warning("Using DeepSpeed")

    model = transformers.AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, **model_kwargs)
    
    # Configure LoRA parameters
    lora_param = {}
    if model_args.peft_type == "dora":
        lora_param["use_dora"] = True
        logger.warning("Using DORA")
    if model_args.peft_type == 'pissa':
        lora_param["init_lora_weights"] = "pissa_niter_4"
        logger.warning("Using PISSA")
    if model_args.peft_type == 'rslora':
        lora_param["use_rslora"] = True
        logger.warning("Using RSLORA")
        
    config = LoraConfig(
        r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        target_modules=get_target(model.config.model_type.lower(), model.named_modules()),
        inference_mode=False,
        lora_dropout=model_args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        **lora_param
    )

    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    if torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        use_fast=False,
    )
    
    # Add special tokens if needed
    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN
    
    if special_tokens_dict:
        tokenizer.add_special_tokens(special_tokens_dict)
        model.resize_token_embeddings(len(tokenizer))
    
    return model, tokenizer


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


class TaskDataset(Dataset):
    """Dataset for task-specific fine-tuning."""

    def __init__(self, 
                 data_path: str, 
                 tokenizer: transformers.PreTrainedTokenizer
                 ):
        super(TaskDataset, self).__init__()
        logger.warning(f"Loading data from {data_path}...")
        
        with open(data_path, 'r') as f:
            list_data_dict = json.load(f)
        
        logger.warning(f"Loaded {len(list_data_dict)} examples")
        
        prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
        
        sources = [
            prompt_input.format_map(example) if example.get("input", "") != "" 
            else prompt_no_input.format_map(example) 
            for example in list_data_dict
        ]
        
        targets = [f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict]

        logger.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer)
        
        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, 
    data_args: DataArguments
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = TaskDataset(
        tokenizer=tokenizer,
        data_path=data_args.train_data_path
    )
    
    eval_dataset = None
    if data_args.eval_data_path is not None:
        eval_dataset = TaskDataset(
            tokenizer=tokenizer,
            data_path=data_args.eval_data_path
        )
    
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    
    return dict(
        train_dataset=train_dataset, 
        eval_dataset=eval_dataset, 
        data_collator=data_collator
    )


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # Set output directory based on domain and task
    if data_args.domain and data_args.task:
        output_dir = os.path.join(training_args.output_dir, data_args.domain, data_args.task)
        training_args.output_dir = output_dir
    
    # Create output directory if it doesn't exist
    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir, exist_ok=True)
    
    model, tokenizer = load_model_and_tokenizer(model_args, training_args)
    
    data_module = make_supervised_data_module(
        tokenizer=tokenizer, 
        data_args=data_args
    )
    
    logger.warning("Creating trainer...")
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **data_module
    )
    
    # Training
    if training_args.do_train:
        logger.info(f"Training adapter for domain: {data_args.domain}, task: {data_args.task}...")
        train_result = trainer.train()
        
        # Save adapter
        trainer.save_model(training_args.output_dir)
        logger.info(f"Saved adapter for {data_args.domain}/{data_args.task} successfully")
        
        # Save metrics
        metrics = train_result.metrics
        metrics["domain"] = data_args.domain
        metrics["task"] = data_args.task
        metrics["train_samples"] = len(data_module["train_dataset"])
        
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
    
    # Evaluation
    if training_args.do_eval and data_module["eval_dataset"] is not None:
        logger.info(f"Evaluating adapter for domain: {data_args.domain}, task: {data_args.task}...")
        eval_results = trainer.evaluate()
        
        metrics = eval_results
        metrics["domain"] = data_args.domain
        metrics["task"] = data_args.task
        metrics["eval_samples"] = len(data_module["eval_dataset"])
        
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


def main():
    train()


if __name__ == "__main__":
    main()