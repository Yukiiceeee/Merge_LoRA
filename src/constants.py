PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
    "prompt_mc": (
        "The following are multiple choice questions (with answers) about medicine. "
        "For each question, select ONLY the letter (A, B, C, D, E, F, or G) corresponding to the correct answer."
        "Do not include any explanations, just provide the letter."
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Answer:"
    ),
}

SELECT_PROMPT_START = (
    "Determine which of the following datapoint is in which domain, what is the task.\n"
    "instruction:{instruction}\n"
    "input:{input}\n"
    "output:{output}\n"
)
SELECT_PROMPT_START_TRAIN = (
    "Determine which of the following instruction-input-output datapoint is in which domain, what is the task, and what skills are required.\n"
    "instruction:{instruction}\n"
    "input:{input}\n"
)
SELECT_PROMPT_END = (
    "Select from below list:\n"
    "Domains:{domain_list}\n"
    "Tasks:{task_list}\n"
    "Results:"
)

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"