from cot_eval.evaluation.Evaluator import get_evaluator
import json
from datasets import Dataset


def make_sft_GSM8k():
    # generate CoTs from qwen7b on unused GSM8k set and keep only the correct ones
    dataset_path = "/juice5b/scr5b/kaitwang/evolm-evals/finetune/llama-factory/sft_data/gsm8k_unused_sft.json"
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset_data = json.load(f)
    
    dataset = Dataset.from_dict(dataset_data["data"])
    dataset_info = dataset_data["dataset_info"]
    print(f"Using preserved HF format - Num samples {len(dataset)}")
    print(f"Dataset info: {dataset_info['name']}, Features: {list(dataset_info['column_names'])}")
    
    evaluator = get_evaluator("GSM8k", answer_extraction_form = "both")
    gsm8k_good_CoT = dataset.filter(lambda x: evaluator.check_answers_equiv(evaluator.extract_answer_from_model_completion(x['qwen_7B_solution']), x['gt_solution']))
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

    output_path = "/juice5b/scr5b/kaitwang/evolm-evals/finetune/llama-factory/sft_data/gsm8k_sft_formatted.json"
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(formated_dataset, f)
    
    print(f"saved {len(formated_dataset)} final examples to {output_path}")


if __name__ == "__main__":
    make_sft_GSM8k()




