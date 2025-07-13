from cot_eval.evaluation.Evaluator import get_evaluator
import json


def make_sft_GSM8k():
# generate CoTs from qwen7b on unused set and conduct rejection sampling to keep only the correct ones
    dataset_path = "/Users/kaitlynwang/evolm/finetune/llama-factory/sft_data/gsm8k_unused_sft.json"
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    evaluator = get_evaluator("GSM8k", answer_extraction_form = "both")
    gsm8k_good_CoT = dataset.filter(lambda x: evaluator.check_answers_equiv(evaluator.extract_answer_from_model_completion(x['qwen_7B_solution']), x['gt_solution']))

    
    



