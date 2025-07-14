from cot_eval.utils.data_utils import load_data, split_into_batches
from cot_eval.utils.utils import replace_None, custom_print, fix_seed
from cot_eval.utils.math_utils import avg
from cot_eval.evaluation.Evaluator import get_evaluator

from argparse import ArgumentParser
import os
import json
import torch
import numpy as np
from tqdm import tqdm, trange
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from pathlib import Path
from functools import partial

pprint = partial(custom_print, fname=os.path.basename(__file__), color="blue")

def main(args):
    with open(args.prompt_config_file, "r") as f:
        prompt_config = json.load(f)
    assert "answer_extraction_format" in prompt_config, "`answer_extraction_format` not found in prompt_config"
    
    #! Load ORM if available
    if args.orm_ckpt_dir is not None:
        pprint(f"Loading ORM from {args.orm_ckpt_dir}")
        orm = AutoModelForSequenceClassification.from_pretrained(
            args.orm_ckpt_dir,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            num_labels=1,
        )
        orm_tokenizer = AutoTokenizer.from_pretrained(args.orm_ckpt_dir)
    else:
        orm = None
        orm_tokenizer = None
        
    #! Load PRM if available
    if args.prm_ckpt_dir is not None:
        raise NotImplementedError("PRM not implemented yet")
    else:
        prm = None
        prm_tokenizer = None
    
    test_dataset_names = args.test_dataset_name.split(",")
    for test_dataset_name in test_dataset_names:
        pprint(f"Processing test dataset: {test_dataset_name}...")
        model_out_dir = os.path.join(args.model_output_root, args.examinee_model_id, test_dataset_name, f"chunk_{args.chunk_idx}-{args.total_chunks}")
        if not os.path.exists(model_out_dir):
            pprint(f"Model output directory not found at {model_out_dir}.")
            continue
        
        examinee_results_file = os.path.join(model_out_dir, "examinee_results.json")
        if not os.path.exists(examinee_results_file):
            pprint(f"ERROR: Examinee results not found at {examinee_results_file}.")
            continue
        
        grader_results_file = os.path.join(model_out_dir, "grader_results.json")
        if os.path.exists(grader_results_file) and not args.force:
            try:
                with open(grader_results_file, "r") as f:
                    _tmp = json.load(f)
                pprint(f"Grader results already exist at {grader_results_file}.")
                del _tmp
                continue
            except json.JSONDecodeError:
                pprint(f"Grader results file is corrupted at {grader_results_file}. Re-evaluating...")
                os.remove(grader_results_file)
        
        #! Save args
        args_file = os.path.join(model_out_dir, "args--grader.json")
        with open(args_file, "w") as f:
            json.dump(vars(args), f, indent=4)
        pprint(f"Grader args saved to {args_file}")
        
        #! Load evaluator
        evaluator = get_evaluator(test_dataset_name, prompt_config["answer_extraction_format"])
        pprint(f"Loaded {evaluator.__class__.__name__} for {test_dataset_name}")
            
        #! Load examinee results
        with open(examinee_results_file, "r") as f:
            examinee_results = json.load(f)
        pprint(f"Loaded {len(examinee_results)} examinee results from {examinee_results_file}")
        
        majority_model_solution_correctness_list = []
        correct_model_solution_existence_list = []
        best_orm_score_model_solution_correctness_list = []
        best_prm_score_model_solution_correctness_list = []
        orm_score_mean_list = []
        prm_score_mean_list = []
        updated_examinee_results = []
        num_generations = None
        
        pprint(f"Evaluating {len(examinee_results)} examinee results...")
        with tqdm(total=len(examinee_results)) as pbar:
            #todo: batch processing
            for data in examinee_results:
                id_, problem, gt_solution, gt_answer = data["id"], data["problem"], data["gt_solution"], data["gt_answer"]
                model_responses = data.pop("model_responses")
                assert len(model_responses) > 0, f"No model responses found for problem {id_}"
                
                #! Update `model_responses` with the following new fields
                # 1. `correctness` (`evaluator`'s judgement based on `gt_solution`)
                candidate_solutions = [mr["text"] for mr in model_responses]
                
                if num_generations is None:
                    num_generations = len(candidate_solutions)
                else:
                    assert num_generations == len(candidate_solutions), f"Number of generations mismatch for problem {id_}"
                    
                assert gt_solution is not None, f"Ground-truth solution not found for problem {id_}"
                if gt_answer is None:
                    gt_answer = evaluator.extract_answer_from_gold_solution(gt_solution)
                    data["gt_answer"] = gt_answer
                
                candidate_answers = []
                candidate_correctness = []
                for cs in candidate_solutions:
                    ca = evaluator.extract_answer_from_model_completion(cs)
                    cc = evaluator.check_answers_equiv(ca, gt_answer)
                    candidate_answers.append(ca)
                    candidate_correctness.append(cc)
                
                assert len(candidate_answers) == len(candidate_correctness) == len(model_responses)
                for ca, cc, mr in zip(candidate_answers, candidate_correctness, model_responses):
                    mr["extracted_answer"] = ca
                    mr["correctness"] = cc
                
                # 2. `orm_score` (if ORM is available)
                if orm is not None:
                    def get_chat_prompt(tokenizer, query, response):
                        chat_prompt = [{"role": "user", "content": query}, {"role": "assistant", "content": response}]
                        chat_prompt_tokenized = tokenizer.apply_chat_template(chat_prompt, tokenize=False)
                        return chat_prompt_tokenized
                    
                    prompts = [get_chat_prompt(orm_tokenizer, problem, mr["text"]) for mr in model_responses]
                    inputs = orm_tokenizer(
                        prompts,
                        padding=True,
                        return_tensors="pt",
                    )
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    with torch.no_grad():
                        outputs = orm(**inputs)
                        scores = outputs.logits.view(-1).cpu().float().numpy().tolist()
                    
                    assert len(scores) == len(model_responses)
                    for s, mr in zip(scores, model_responses):
                        mr["orm_score"] = s
                else:
                    for mr in model_responses:
                        mr["orm_score"] = None
                
                # 3. `prm_score` (if PRM is available)
                if prm is not None:
                    raise NotImplementedError("PRM not implemented yet")
                else:
                    for mr in model_responses:
                        mr["prm_score"] = None
                
                #! Update data with evaluation results
                majority_completion, majority_answer, majority_index = evaluator.find_majority_completion_and_answer(candidate_solutions)
                
                correct_answer_existence = any(candidate_correctness)
                correct_answer_ratio = avg(replace_None(candidate_correctness, replacement=False))
                
                if all([mr["orm_score"] is not None for mr in model_responses]):
                    orm_score_mean = avg([mr["orm_score"] for mr in model_responses])
                    best_orm_score_index = np.argmax([mr["orm_score"] for mr in model_responses])
                else:
                    orm_score_mean = None
                    best_orm_score_index = None
                    
                if all([mr["prm_score"] is not None for mr in model_responses]):
                    prm_score_mean = avg([mr["prm_score"] for mr in model_responses])
                    best_prm_score_index = np.argmax([mr["prm_score"] for mr in model_responses])
                else:
                    prm_score_mean = None
                    best_prm_score_index = None
                
                data["majority_correctness"] = model_responses[majority_index]["correctness"]
                data["majority_model_answer"] = model_responses[majority_index]["extracted_answer"]
                data["majority_model_response"] = model_responses[majority_index]
                
                data["best_orm_score_correctness"] = model_responses[best_orm_score_index]["correctness"] if best_orm_score_index is not None else None
                data["best_orm_score_model_answer"] = model_responses[best_orm_score_index]["extracted_answer"] if best_orm_score_index is not None else None
                data["best_orm_score_model_response"] = model_responses[best_orm_score_index] if best_orm_score_index is not None else None
                
                data["best_prm_score_correctness"] = model_responses[best_prm_score_index]["correctness"] if best_prm_score_index is not None else None
                data["best_prm_score_model_answer"] = model_responses[best_prm_score_index]["extracted_answer"] if best_prm_score_index is not None else None
                data["best_prm_score_model_response"] = model_responses[best_prm_score_index] if best_prm_score_index is not None else None
                
                data["correct_answer_existence"] = correct_answer_existence
                data["correct_answer_ratio"] = correct_answer_ratio
                
                data["orm_score_mean"] = orm_score_mean
                data["prm_score_mean"] = prm_score_mean
                
                data["model_responses"] = model_responses
                
                #! Record statistics
                majority_model_solution_correctness_list.append(model_responses[majority_index]["correctness"])
                correct_model_solution_existence_list.append(correct_answer_existence)
                best_orm_score_model_solution_correctness_list.append(model_responses[best_orm_score_index]["correctness"] if best_orm_score_index is not None else None)
                best_prm_score_model_solution_correctness_list.append(model_responses[best_prm_score_index]["correctness"] if best_prm_score_index is not None else None)
                orm_score_mean_list.append(orm_score_mean)
                prm_score_mean_list.append(prm_score_mean)
                
                #! Record simplified results
                updated_examinee_results.append(data)
        
                pbar.update(1)
        
        #! Save overall evaluation results
        majority_voting_acc = avg(replace_None(majority_model_solution_correctness_list, replacement=False))
        pass_at_k_acc = avg(replace_None(correct_model_solution_existence_list, replacement=False))
        best_orm_acc = avg(replace_None(best_orm_score_model_solution_correctness_list, replacement=False)) if orm is not None else None
        best_prm_acc = avg(replace_None(best_prm_score_model_solution_correctness_list, replacement=False)) if prm is not None else None
        avg_orm_score = avg(replace_None(orm_score_mean_list, replacement=0.0)) if orm is not None else None
        avg_prm_score = avg(replace_None(prm_score_mean_list, replacement=0.0)) if prm is not None else None
        
        pprint("=" * 42)
        pprint(f"Overall evaluation results on {test_dataset_name}")
        pprint("-" * 42)
        pprint(f"Maj@{num_generations} acc: {majority_voting_acc}")
        pprint(f"Pass@{num_generations} acc: {pass_at_k_acc}")
        pprint(f"Best ORM score acc: {best_orm_acc}")
        pprint(f"Best PRM score acc: {best_prm_acc}")
        pprint(f"Average ORM score: {avg_orm_score}")
        pprint(f"Average PRM score: {avg_prm_score}")
        pprint("=" * 42)
        
        grader_results = {
            "num_generations": num_generations,
            "majority_voting_acc": majority_voting_acc,
            "pass_at_k_acc": pass_at_k_acc,
            "best_orm_acc": best_orm_acc,
            "best_prm_acc": best_prm_acc,
            "avg_orm_score": avg_orm_score,
            "avg_prm_score": avg_prm_score,
            "updated_examinee_results": updated_examinee_results,
        }
        with open(grader_results_file, "w") as f:
            json.dump(grader_results, f)
        
        grader_results_overview_file = os.path.join(model_out_dir, "grader_results--overview.json")
        with open(grader_results_overview_file, "w") as f:
            json.dump({
                "num_problems": len(updated_examinee_results),
                "num_generations_per_problem": num_generations,
                "majority_voting_acc": majority_voting_acc,
                "pass_at_k_acc": pass_at_k_acc,
                "best_orm_acc": best_orm_acc,
                "best_prm_acc": best_prm_acc,
                "avg_orm_score": avg_orm_score,
                "avg_prm_score": avg_prm_score,
            }, f, indent=4)
        
        pprint(f"Grader results saved to {grader_results_file} and {grader_results_overview_file}")
    

if __name__ == "__main__":
    parser = ArgumentParser()
    
    # input
    parser.add_argument("--model_output_root", type=str, default="outputs")
    parser.add_argument("--test_dataset_name", type=str, required=True)
    parser.add_argument("--examinee_model_id", type=str, required=True)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--total_chunks", type=int, default=1)
    
    # misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    
    # ORM
    parser.add_argument("--orm_ckpt_dir", type=str, default=None)
    
    # PRM
    parser.add_argument("--prm_ckpt_dir", type=str, default=None)
    
    # prompt
    parser.add_argument("--prompt_config_file", type=str, required=True)
    
    args = parser.parse_args()
    
    assert not args.model_output_root.endswith("/")
    if args.seed != 42:
        args.model_output_root = args.model_output_root + f"-seed{args.seed}"
    
    pprint(f"Grader args:\n{'-' * 42}\n{json.dumps(vars(args), indent=4)}\n{'-' * 42}")
        
    fix_seed(args.seed)
    main(args)
