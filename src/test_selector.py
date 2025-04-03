from transformers import AutoModelForCausalLM, set_seed
import transformers
from peft import PeftModel, PeftConfig
import torch
import json
import fire
import re
from tqdm import tqdm
import logging
from utils import load_data_to_list, smart_tokenizer_and_embedding_resize, save_data_to_json
import os
from constants import (
    IGNORE_INDEX,
    DEFAULT_PAD_TOKEN,
    DEFAULT_EOS_TOKEN,
    DEFAULT_BOS_TOKEN,
    DEFAULT_UNK_TOKEN,
    SELECT_PROMPT_START,
    SELECT_PROMPT_END
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)

def initialize_counters(data_dict):
    counters = {'total': {'all': 0}, 'domains': {}}

    for domain, tasks in data_dict.items():
        counters['domains'][domain] = {'all': 0}
        for task in tasks.keys():
            counters['domains'][domain][task] = 0

    return counters
    
def run(
    model_path: str = "/d2/mxy/Models/Meta-Llama-3-8B",
    selector_path: str = "/d2/mxy/TASA/selector/selector_test",
    test_data: str = "/d2/mxy/TASA/data/test/test_exp1_982.json",
    data_config_path:str = "/d2/mxy/TASA/config/data_config_exp_6.json"
):
    if re.search(r'checkpoint-\d+$', selector_path):
        j = json.load(open(f"{os.path.dirname(selector_path)}/all_results.json", 'r'))
    else:
        j = json.load(open(f"{selector_path}/all_results.json", 'r'))
    if 'test_prompt' in j:
        PROMPT = j['test_prompt']
    else:
        PROMPT = j['prompt']
    model = transformers.AutoModelForCausalLM.from_pretrained(
            model_path, device_map='auto', torch_dtype='auto', trust_remote_code=True)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, use_fast=False)
    
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

    model = PeftModel.from_pretrained(model, selector_path)
    with open(test_data, 'r') as f:
        data = json.load(f)
    with open(data_config_path, 'r') as f:
        data_config = json.load(f)
    counters = initialize_counters(data_config)
    # Iterate over data with a progress bar
    for i in tqdm(data):
        input_prompt = PROMPT.format_map(i)
        model_input = tokenizer(input_prompt, return_tensors="pt").to("cuda")
        output = model.generate(**model_input, max_new_tokens=25)
        out_text = tokenizer.decode(output[0], skip_special_tokens=True).removeprefix(input_prompt)
        str_list = out_text.split("\n")

        # Skip if the output does not contain enough lines
        if len(str_list) < 2:
            continue

        # Extract domain and task from the output
        domain = str_list[0].removeprefix("domain:")
        task = str_list[1].removeprefix("task:")
        
        print(f"domain: {domain}, task: {task}")

        # Update counters if the domain and task match
        if domain == i['domain'] and task == i['task']:
            counters['total']['all'] += 1
            counters['domains'][domain]['all'] += 1
            counters['domains'][domain][task] += 1


    # Print results
    total_count = counters['total']['all']
    data_length = len(data)
    counters['acc'] = total_count / data_length
    print(f"acc: {total_count / data_length}")

    for domain, tasks in counters['domains'].items():
        print(f"{domain}: {tasks['all']}, {', '.join([f'{k}: {v}' for k, v in tasks.items() if k != 'all'])}")
    
    with open(f"{selector_path}/acc.json", 'w') as f:
        json.dump(counters, f, indent=4)


if "__main__" == __name__:
    fire.Fire(run)


# from transformers import AutoModelForCausalLM, set_seed
# import transformers
# from peft import PeftModel, PeftConfig
# import torch
# import json
# import fire
# import re
# from tqdm import tqdm
# import logging
# import os

# logger = logging.getLogger(__name__)
# logger.setLevel(logging.ERROR)

# def initialize_counters(data_dict):
#     counters = {'total': {'all': 0}, 'domains': {}}

#     for domain, tasks in data_dict.items():
#         counters['domains'][domain] = {'all': 0}
#         for task in tasks.keys():
#             counters['domains'][domain][task] = 0

#     return counters
    
# def run(
#     model_path: str = "/d2/mxy/Models/Meta-Llama-3-8B",
#     selector_path: str = "/d2/mxy/TASA/selector/selector1",
#     test_data: str = "/d2/mxy/TASA/data/test/test_exp1_982.json",
#     data_config_path:str = "/d2/mxy/TASA/config/data_config_exp_6.json"
# ):
#     if re.search(r'checkpoint-\d+$', selector_path):
#         j = json.load(open(f"{os.path.dirname(selector_path)}/all_results.json", 'r'))
#     else:
#         j = json.load(open(f"{selector_path}/all_results.json", 'r'))
#     if 'test_prompt' in j:
#         PROMPT = j['test_prompt']
#     else:
#         PROMPT = j['prompt']
    
#     # 1. 加载tokenizer
#     tokenizer = transformers.AutoTokenizer.from_pretrained(
#         model_path,
#         trust_remote_code=True,
#         use_fast=False,
#     )
    
#     # 2. 添加特殊token（与训练时一致）
#     special_tokens_dict = dict()
#     if tokenizer.pad_token is None:
#         special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
#     if tokenizer.eos_token is None:
#         special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
#     if tokenizer.bos_token is None:
#         special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
#     if tokenizer.unk_token is None:
#         special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

#     # 3. 加载模型
#     model = transformers.AutoModelForCausalLM.from_pretrained(
#         model_path, 
#         device_map='auto', 
#         torch_dtype='auto', 
#         trust_remote_code=True
#     )
    
#     # 4. 调整词表大小
#     smart_tokenizer_and_embedding_resize(
#         special_tokens_dict=special_tokens_dict,
#         tokenizer=tokenizer,
#         model=model,
#     )
    
#     # 5. 加载selector
#     model = PeftModel.from_pretrained(model, selector_path)
#     with open(test_data, 'r') as f:
#         data = json.load(f)
#     with open(data_config_path, 'r') as f:
#         data_config = json.load(f)
#     counters = initialize_counters(data_config)
#     # Iterate over data with a progress bar
#     for i in tqdm(data):
#         input_prompt = PROMPT.format_map(i)
#         model_input = tokenizer(input_prompt, return_tensors="pt").to("cuda")
#         output = model.generate(**model_input, max_new_tokens=25)
#         out_text = tokenizer.decode(output[0], skip_special_tokens=True).removeprefix(input_prompt)
#         str_list = out_text.split("\n")

#         # Skip if the output does not contain enough lines
#         if len(str_list) < 2:
#             continue

#         # Extract domain and task from the output
#         domain = str_list[0].removeprefix("domain:")
#         task = str_list[1].removeprefix("task:")
        
#         print(f"domain: {domain}, task: {task}")

#         # Update counters if the domain and task match
#         if domain == i['domain'] and task == i['task']:
#             counters['total']['all'] += 1
#             counters['domains'][domain]['all'] += 1
#             counters['domains'][domain][task] += 1


#     # Print results
#     total_count = counters['total']['all']
#     data_length = len(data)
#     counters['acc'] = total_count / data_length
#     print(f"acc: {total_count / data_length}")

#     for domain, tasks in counters['domains'].items():
#         print(f"{domain}: {tasks['all']}, {', '.join([f'{k}: {v}' for k, v in tasks.items() if k != 'all'])}")
    
#     with open(f"{selector_path}/acc.json", 'w') as f:
#         json.dump(counters, f, indent=4)


# if "__main__" == __name__:
#     fire.Fire(run)