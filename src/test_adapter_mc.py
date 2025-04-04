import os
import json
import torch
import logging
import argparse
from tqdm import tqdm
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import load_data_to_list, smart_tokenizer_and_embedding_resize, save_data_to_json
import re
from constants import (
    IGNORE_INDEX,
    DEFAULT_PAD_TOKEN,
    DEFAULT_EOS_TOKEN,
    DEFAULT_BOS_TOKEN,
    DEFAULT_UNK_TOKEN,
    SELECT_PROMPT_START,
    SELECT_PROMPT_END
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def evaluate_adapter(
    model_path: str,
    adapter_path: str,
    test_data_path: str,
    use_base_model: bool = False
):
    """
    评估adapter的效果或原始模型的效果
    
    Args:
        model_path: 基础模型路径
        adapter_path: 训练好的adapter路径
        test_data_path: 测试数据路径
        use_base_model: 是否只使用基础模型（不加载adapter）
    """
    # 检查文件是否存在
    if not os.path.exists(test_data_path):
        raise FileNotFoundError(f"测试数据文件不存在: {test_data_path}")
    if not use_base_model and not os.path.exists(adapter_path):
        raise FileNotFoundError(f"Adapter模型路径不存在: {adapter_path}")
    
    # 确定结果保存路径
    save_dir = model_path.split('/')[-1] if use_base_model else adapter_path
    # 如果是基础模型评估，创建保存结果的目录
    if use_base_model:
        # 从测试数据路径推断任务信息
        # 假设格式为 .../data_adapters/domain/task/test.json
        path_parts = test_data_path.split('/')
        task_idx = path_parts.index('test.json') - 1
        domain_idx = task_idx - 1
        
        domain = path_parts[domain_idx] if domain_idx >= 0 and domain_idx < len(path_parts) else "unknown"
        task = path_parts[task_idx] if task_idx >= 0 and task_idx < len(path_parts) else "unknown"
        
        # 为基础模型创建保存目录
        base_model_name = os.path.basename(model_path)
        save_dir = os.path.join(os.path.dirname(adapter_path), f"base_{base_model_name}", domain, task)
        os.makedirs(save_dir, exist_ok=True)
        
        # 创建一个简单的task_info.json
        task_info = {
            "domain": domain,
            "task": task,
            "use_few_shot": False
        }
    else:
        # 加载任务信息
        task_info_path = os.path.join(adapter_path, "task_info.json")
        task_info = {}
        if os.path.exists(task_info_path):
            with open(task_info_path, "r") as f:
                task_info = json.load(f)
            logger.info(f"加载任务信息: {task_info}")
    
    # 加载基础模型和tokenizer
    logger.info(f"加载基础模型: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        device_map="auto", 
        torch_dtype="auto", 
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, 
        trust_remote_code=True
    )

    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )
    
    # 加载adapter (如果不使用基础模型)
    if not use_base_model:
        logger.info(f"加载adapter: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
    else:
        logger.info("使用基础模型进行评估，不加载adapter")
    
    # 加载测试数据
    logger.info(f"加载测试数据: {test_data_path}")
    with open(test_data_path, "r") as f:
        test_data = json.load(f)
    
    # 准备提示模板
    domain = task_info.get("domain", "")
    task = task_info.get("task", "")
    use_few_shot = task_info.get("use_few_shot", False)
    
    # 为多选题创建特定的提示模板
    if task == "mc":
        base_prompt = """The following are multiple choice questions (with answers) about medicine. 
For each question, select ONLY the letter (A, B, C, D, E, F, or G) corresponding to the correct answer.
Do not include any explanations, just provide the letter.

{instruction}
{input}
Answer: """

        few_shot_examples = """
Example 1:
Question: Beta-blockers are most commonly used to treat which condition?
A. Asthma
B. Hypertension
C. Hypothyroidism
D. Diabetes mellitus
Answer: B

Example 2:
Question: Which organ is primarily responsible for detoxification of drugs in the body?
A. Kidney
B. Lungs
C. Liver
D. Spleen
E. Pancreas
Answer: C

Now answer the following question:
"""

        prompt_template = base_prompt + few_shot_examples if use_few_shot else base_prompt
    else:
        # 默认提示模板
        prompt_input = """### Instruction:
{instruction}

### Input:
{input}

### Response:
"""
        prompt_no_input = """### Instruction:
{instruction}

### Response:
"""
        prompt_template = None  # 将在循环中根据样本决定
    
    # 创建结果列表
    predictions = []
    
    # 设置生成参数
    gen_kwargs = {
        "max_new_tokens": 32,      # 对于选择题，答案通常很短
        "temperature": 0.1,        # 使用较低温度以获得更确定的答案
        "top_p": 0.9,
        "do_sample": True
    }
    
    # 评估模型
    logger.info("开始评估...")
    correct_count = 0
    
    # 对测试数据进行推理
    for example in tqdm(test_data):
        # 构建输入提示
        if task == "mc":
            input_prompt = prompt_template.format_map(example)
        else:
            if example.get("input", "") != "":
                input_prompt = prompt_input.format_map(example)
            else:
                input_prompt = prompt_no_input.format_map(example)
        
        # 标记化
        inputs = tokenizer(input_prompt, return_tensors="pt").to(model.device)
        
        # 生成输出
        with torch.no_grad():
            output = model.generate(**inputs, **gen_kwargs)
        
        # 解码输出
        full_output = tokenizer.decode(output[0], skip_special_tokens=True)
        pred_text = full_output[len(input_prompt):].strip()
        
        # 保存预测结果
        example_copy = example.copy()
        example_copy["pred"] = pred_text
        predictions.append(example_copy)
        
        # 处理并检查正确性
        if task == "mc":
            # 对多选题，我们只关心第一个字母(A/B/C/D)
            # 有时模型会输出"The answer is A"等格式，需要处理
            first_char = pred_text[:1] if pred_text else ""
            if first_char == example["output"] or pred_text.startswith(example["output"]):
                correct_count += 1
        else:
            # 对其他任务，检查完全匹配或前缀匹配
            if pred_text == example["output"] or pred_text.startswith(example["output"]) or example["output"].startswith(pred_text):
                correct_count += 1
    
    # 计算准确率
    accuracy = correct_count / len(test_data) if test_data else 0
    model_type = "基础模型" if use_base_model else "适配器模型"
    logger.info(f"评估完成! {model_type}准确率: {accuracy:.4f} ({correct_count}/{len(test_data)})")
    
    # 保存预测结果
    pred_path = os.path.join(save_dir, "pred.json")
    with open(pred_path, "w") as f:
        json.dump(predictions, f, indent=4)
    logger.info(f"预测结果已保存至: {pred_path}")
    
    # 保存准确率
    acc_path = os.path.join(save_dir, "acc.json")
    acc_result = {
        "accuracy": accuracy,
        "correct_count": correct_count,
        "total_count": len(test_data),
        "domain": domain,
        "task": task,
        "model_type": "base_model" if use_base_model else "adapter_model"
    }
    with open(acc_path, "w") as f:
        json.dump(acc_result, f, indent=4)
    logger.info(f"准确率结果已保存至: {acc_path}")
    
    return accuracy

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="评估adapter模型或基础模型的效果")
    parser.add_argument("--model_path", default="/d2/mxy/Models/Meta-Llama-3-8B", type=str, help="基础模型路径")
    parser.add_argument("--adapter_path", default="/d2/mxy/TASA/models/adapters/med/mc", type=str, help="adapter模型路径")
    parser.add_argument("--test_data_path", default="/d2/mxy/TASA/data/data_adapters/med/mc/test.json", type=str, help="测试数据路径")
    parser.add_argument("--use_base_model", action="store_true", help="是否只使用基础模型（不加载adapter）")
    
    args = parser.parse_args()
    
    evaluate_adapter(
        model_path=args.model_path,
        adapter_path=args.adapter_path,
        test_data_path=args.test_data_path,
        use_base_model=args.use_base_model
    )