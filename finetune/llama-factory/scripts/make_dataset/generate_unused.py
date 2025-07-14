import os
import sys

sys.path.append('evaluation/cot-eval-harness')

os.environ['HF_HOME'] = '/juice5b/scr5b/kaitwang/cache/huggingface'

from datasets import load_dataset
from cot_eval.evaluation.Evaluator import get_evaluator
import json

def generate_unused_gsm8k():

    # print("loading used dataset")
    # gsm8k_used = load_dataset("ZhentingNLP/mathaug", split="gsm8k_sampled_250k")
    print("loading full dataset")
    gsm8k_full = load_dataset("ZhentingNLP/mathaug", split="gsm8k")

    # print("creating filter")
    # used_ids = set(example['unique_id'] for example in gsm8k_used)
    # gsm8k_unused = gsm8k_full.filter(lambda x: x['unique_id'] not in used_ids)
    # print(f"{len(gsm8k_unused)} unused samples")

    evaluator = get_evaluator("GSM8k", answer_extraction_format = "both")
    gsm8k_good_CoT = gsm8k_full.filter(lambda x: evaluator.check_answers_equiv(evaluator.extract_answer_from_model_completion(x['qwen_7B_solution']), x['gt_solution']))
    print(f"Num good samples {len(gsm8k_good_CoT)}")

    instruction_string = """Solve the following problem efficiently and clearly:

- For simple problems (2 steps or fewer):
Provide a concise solution with minimal explanation.

- For complex problems (3 steps or more):
Use this step-by-step format:

## Step 1: [Concise description]
[Brief explanation and calculations]

## Step 2: [Concise description]
[Brief explanation and calculations]

...

Regardless of the approach, always conclude with:

Therefore, the final answer is: $\\boxed{answer}$. I hope it is correct.

Where [answer] is just the final number or expression that solves the problem."""

    formated_dataset = [{"instruction": instruction_string, "input": d["problem"], "output": d["qwen_7B_solution"]} for d in gsm8k_good_CoT]

    output_path = "sft_data/gsm8k_sft_all280k_formatted.json"
    print(f"saved {len(formated_dataset)} final examples to {output_path}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(formated_dataset, f) 

if __name__ == "__main__":
    generate_unused_gsm8k()


