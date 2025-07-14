from cot_eval.utils.data_utils import load_data, split_into_batches
from cot_eval.utils.utils import custom_print, fix_seed

from argparse import ArgumentParser
import os
import json
import torch
import numpy as np
import random
from tqdm import tqdm, trange
from vllm import LLM, SamplingParams
from time import time
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Optional
from functools import partial

pprint = partial(custom_print, fname=os.path.basename(__file__), color="green")
device = torch.device('cuda')

def calculate_ground_truth_log_probs(hf_model, hf_tokenizer, prompts: list[str], ground_truth_solutions:list[str]):
        log_probs = []
        for prompt, gt_CoT in zip(prompts, ground_truth_solutions):
            full_text = prompt + gt_CoT
            inputs = hf_tokenizer(full_text, return_tensors="pt")
            inputs = {key: value.to(device) for key, value in inputs.items()}
            with torch.no_grad():
                outputs = hf_model(**inputs)
                prompt_len = len(hf_tokenizer(prompt, return_tensors="pt").input_ids[0])
                gt_log_probs = outputs.logits[0, prompt_len-1:-1]  # -1 to predict the next token
                log_prob = gt_log_probs.log_softmax(-1).gather(-1, inputs["input_ids"][0, prompt_len:].unsqueeze(-1)).sum()
                log_probs.append(log_prob.item())
        return log_probs



def load_model(model_ckpt_dir, api, max_model_len, tensor_parallel_size, model_seed):
    if api == "vllm":
        model = LLM(
            seed=model_seed,
            model=model_ckpt_dir, 
            max_model_len=max_model_len,
            gpu_memory_utilization=0.8,
            tensor_parallel_size=tensor_parallel_size,
            enable_prefix_caching=True,
        )

    else:
        raise NotImplementedError(f"API {api} not supported")
    
    return model

def load_hf_model(model_ckpt_dir, model_hf_path, api, max_model_len, tensor_parallel_size, model_seed):
    if api == "hf":
        torch.manual_seed(model_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(model_seed)
        
        if os.path.exists(model_ckpt_dir):
            model = AutoModelForCausalLM.from_pretrained(
                model_ckpt_dir,
                torch_dtype="auto",
                device_map="auto"
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_hf_path,
                torch_dtype="auto",
                device_map="auto",
                trust_remote_code=True
            )
        
        model.to(device)
    else:
        raise NotImplementedError(f"API {api} not supported")
    
    return model
    

def prepare_prompt(
    question: str,
    prompt_config: dict,
    apply_chat_template: bool,
    tokenizer: Optional[AutoTokenizer],
    fewshot_examples: list[dict] = [],
    chat_template: Optional[dict] = None,
):
    input_wrapper = prompt_config["input_wrapper"]
    assert "<<<input>>>" in input_wrapper
    
    system_prompt = prompt_config["system_prompt"]
    if apply_chat_template:
        if chat_template is not None:
            user_message_template = chat_template["user"]
            assert "<<<wrapped_input>>>" in user_message_template
            
            assistant_message_template = chat_template["assistant"]
            assert "<<<output>>>" in assistant_message_template
            
            prompt = system_prompt if system_prompt is not None else ""
            for example in fewshot_examples:
                wrapped_input = input_wrapper.replace("<<<input>>>", example["input"])
                prompt += user_message_template.replace("<<<wrapped_input>>>", wrapped_input)
                prompt += assistant_message_template.replace("<<<output>>>", example["output"])
            wrapped_input = input_wrapper.replace("<<<input>>>", question)
            prompt += user_message_template.replace("<<<wrapped_input>>>", wrapped_input)
            prompt += assistant_message_template.split("<<<output>>>")[0]
        else:
            messages = [system_prompt] if system_prompt is not None else []
            for example in fewshot_examples:
                wrapped_input = input_wrapper.replace("<<<input>>>", example["input"])
                messages.append({"role": "user", "content": wrapped_input})
                messages.append({"role": "assistant", "content": example["output"]})
            wrapped_input = input_wrapper.replace("<<<input>>>", question)
            messages.append({"role": "user", "content": wrapped_input})
            
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
    else:
        prompt = system_prompt + "\n\n" if system_prompt is not None else ""
        prompt += input_wrapper.replace("<<<input>>>", question)
    
    return prompt


def generate_responses(
    api,
    batch_prompt: list[str],
    model,
    tokenizer,
    **gen_kwargs
):
    assert isinstance(batch_prompt, list) and len(batch_prompt) > 0
    
    if api == "vllm":
        assert isinstance(model, LLM)
        sampling_params = SamplingParams(
            n=gen_kwargs["num_generations"],
            max_tokens=gen_kwargs["max_new_tokens"],
            repetition_penalty=gen_kwargs["repetition_penalty"],
            temperature=gen_kwargs["temperature"],
            stop=gen_kwargs["stop_tokens"],
            bad_words=gen_kwargs["bad_words"],
            skip_special_tokens=gen_kwargs["skip_special_tokens"]
        )
        batch_output = model.generate(
            prompts=batch_prompt, 
            sampling_params=sampling_params,
            use_tqdm=True
        )
    elif api == "hf":
        raise NotImplementedError("HF API not supported")
    
        model_inputs = tokenizer(batch_prompt, return_tensors="pt", padding=True)
        model_inputs = {key: value.to(device) for key, value in model_inputs.items()}
        
        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=gen_kwargs["max_new_tokens"]
            )
        
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
    else:
        raise NotImplementedError(f"API {api} not supported")
    
    assert len(batch_output) == len(batch_prompt)
    return batch_output


def main(args):
    #! Load prompt config
    pprint(f"Loading prompt config from {args.prompt_config_file}...")
    with open(args.prompt_config_file, "r") as f:
        prompt_config = json.load(f)
    assert "stop_tokens" in prompt_config
    pprint(f"Loaded prompt config:\n{'-' * 42}\n{json.dumps(prompt_config, indent=4)}\n{'-' * 42}\n")
    
    #! Load fewshot examples
    if args.fewshot_examples_file is not None:
        pprint(f"Loading fewshot examples from {args.fewshot_examples_file}...")
        with open(args.fewshot_examples_file, "r") as f:
            fewshot_examples = json.load(f)
        pprint(f"Loaded {len(fewshot_examples)} fewshot examples")
        pprint(f"Fewshot examples:\n{'-' * 42}\n{json.dumps(fewshot_examples, indent=4)}\n{'-' * 42}")
    else:
        fewshot_examples = []
        
    #! Load custom chat template
    if args.chat_template_file is not None:
        pprint(f"Loading chat template from {args.chat_template_file}...")
        with open(args.chat_template_file, "r") as f:
            chat_template = json.load(f)
        pprint(f"Loaded custom chat template:\n{'-' * 42}\n{chat_template}\n{'-' * 42}\n")
    else:
        chat_template = None
    
    #! Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_ckpt_dir, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    pprint(f"Loaded tokenizer from {args.model_ckpt_dir}")
    
    #! Load model
    pprint(f"Loading model from {args.model_ckpt_dir}...")
    model = load_model(args.model_ckpt_dir, args.api, args.max_model_len, args.tensor_parallel_size, args.seed)

    hf_model = None
    hf_tokenizer = None
    if args.model_hf_ckpt_dir is not None or args.model_hf_path is not None:
        pprint(f"Loading hf model from {args.model_hf_ckpt_dir, args.model_hf_path}...")
        hf_model = load_hf_model(args.model_hf_ckpt_dir, args.model_hf_path, "hf", args.max_model_len, args.tensor_parallel_size, args.seed)
        hf_tokenizer = AutoTokenizer.from_pretrained(args.model_hf_ckpt_dir if args.model_hf_ckpt_dir else args.model_hf_path, padding_side="left")
        hf_tokenizer.pad_token = hf_tokenizer.eos_token
        hf_tokenizer.pad_token_id = hf_tokenizer.eos_token_id
    
    #! Prepare generation kwargs
    gen_kwargs = {
        "num_generations": args.num_generations,
        "max_new_tokens": args.max_new_tokens,
        "repetition_penalty": args.repetition_penalty,
        "temperature": args.temperature,
        "stop_tokens": prompt_config.get("stop_tokens", []),
        "bad_words": prompt_config.get("bad_words", None),
        "skip_special_tokens": False,
    }
    
    test_dataset_names = args.test_dataset_name.split(",")
    for test_dataset_name in test_dataset_names:
        pprint(f"Processing test dataset: {test_dataset_name}")
        test_dataset_file = os.path.join(args.data_root, "cooked", f"{test_dataset_name}.jsonl")
        if not os.path.exists(test_dataset_file):
            pprint(f"ERROR: Test dataset file not found at {test_dataset_file}!")
            continue
        
        out_dir = os.path.join(args.out_root, args.model_id_for_saving, test_dataset_name, f"chunk_{args.chunk_idx}-{args.total_chunks}")
        os.makedirs(out_dir, exist_ok=True)
        examinee_results_file = os.path.join(out_dir, "examinee_results.json")
        if os.path.exists(examinee_results_file) and not args.force:
            try:
                with open(examinee_results_file, "r") as f:
                    _tmp = json.load(f)
                pprint(f"Examinee results already exist at {examinee_results_file}.")
                del _tmp
                continue
            except json.JSONDecodeError:
                pprint(f"Examinee results file is corrupted at {examinee_results_file}. Re-generating examinee results...")
                os.remove(examinee_results_file)
        
        #! Save args
        args_file = os.path.join(out_dir, "args--examinee.json")
        with open(args_file, "w") as f:
            json.dump(vars(args), f, indent=4)
        pprint(f"Examinee args saved to {args_file}")
        
        #! Load test dataset
        pprint(f"Loading test dataset from {test_dataset_file}...")
        data = load_data(test_dataset_file, "jsonl")
        chunk_size = (len(data) + args.total_chunks - 1) // args.total_chunks
        data = data[chunk_size * args.chunk_idx : chunk_size * (args.chunk_idx + 1)]
        
        id_list = [d["id"] for d in data]
        problem_list = [d["problem"] for d in data]
        prompt_list = [
            prepare_prompt(
                question=d["problem"],
                prompt_config=prompt_config,
                apply_chat_template=args.apply_chat_template,
                fewshot_examples=fewshot_examples,
                tokenizer=tokenizer,
                chat_template=chat_template
            ) for d in data
        ]
        pprint(f"Example prompt:\n{'-' * 42}\n{prompt_list[0]}\n{'-' * 42}")
        solution_list = [d["gt_solution"] for d in data]
        answer_list = [d["gt_answer"] for d in data]
        id_as_batches = split_into_batches(id_list, args.batch_size)
        problem_as_batches = split_into_batches(problem_list, args.batch_size)
        prompt_as_batches = split_into_batches(prompt_list, args.batch_size)
        solution_as_batches = split_into_batches(solution_list, args.batch_size)
        answer_as_batches = split_into_batches(answer_list, args.batch_size)
        assert len(id_as_batches) == len(problem_as_batches) == len(prompt_as_batches) == len(solution_as_batches) == len(answer_as_batches)
        num_batches = len(id_as_batches)
        
        pprint(f"Loaded {len(data)} data points in {num_batches} batches")
        
        #! Evaluate
        pprint('-' * 42)
        pprint(f"Model is taking exam: {test_dataset_name}")
        pprint('-' * 42)
        results = []
        with tqdm(total=num_batches, disable=True) as pbar:
            for batch_idx in range(num_batches):
                batch_id = id_as_batches[batch_idx]
                batch_problem = problem_as_batches[batch_idx]
                batch_prompt = prompt_as_batches[batch_idx]
                batch_solution = solution_as_batches[batch_idx]
                batch_answer = answer_as_batches[batch_idx]
                
                # Collect responses
                batch_candidates = []
                batch_token_cnts = []
                batch_output = generate_responses(
                    args.api,
                    batch_prompt,
                    model,
                    tokenizer,
                    **gen_kwargs
                )
                for output in batch_output:
                    assert len(output.outputs) == gen_kwargs["num_generations"]
                    candidates = [o.text for o in output.outputs]
                    token_cnts = [len(o.token_ids) for o in output.outputs]
                    batch_candidates.append(candidates)
                    batch_token_cnts.append(token_cnts)
                assert len(batch_candidates) == len(batch_token_cnts) == len(batch_id)
                
                if hf_model is not None:
                    gt_log_probs = calculate_ground_truth_log_probs(hf_model, hf_tokenizer, batch_prompt, batch_solution)
                else:
                    gt_log_probs = [None] * len(batch_id)
                
                # Save responses
                for i in range(len(batch_id)):
                    id_ = batch_id[i]
                    problem = batch_problem[i]
                    prompt = batch_prompt[i]
                    solution = batch_solution[i]
                    answer = batch_answer[i]
                    candidates = batch_candidates[i]
                    token_cnts = batch_token_cnts[i]
                    gt_log_prob = gt_log_probs[i]
                    assert len(candidates) == len(token_cnts) == args.num_generations
                    js_dict = {
                        "id": id_,
                        "problem": problem,
                        "prompt": prompt,
                        "gt_solution": solution,
                        "gt_answer": answer,
                        "gt_log_prob": gt_log_prob,
                        "model_responses": [{"text": c, "num_tokens": t} for c, t in zip(candidates, token_cnts)]
                    }
                    results.append(js_dict)

                pbar.update(1)

        with open(examinee_results_file, "w") as f:
            json.dump(results, f)
            
        pprint(f"Results saved to {examinee_results_file}")


if __name__ == "__main__":
    parser = ArgumentParser()

    # misc
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--out_root", type=str, default="outputs")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")

    # data
    parser.add_argument("--test_dataset_name", type=str, required=True)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--total_chunks", type=int, default=1)
    
    # prompt
    parser.add_argument("--apply_chat_template", action="store_true")
    parser.add_argument("--chat_template_file", type=str, default=None)
    parser.add_argument("--prompt_config_file", type=str, required=True)
    parser.add_argument("--fewshot_examples_file", type=str, default=None)
    
    # llm
    parser.add_argument("--api", type=str, choices=["vllm", "hf"], default="vllm")
    parser.add_argument("--model_ckpt_dir", type=str, required=True)
    parser.add_argument("--model_hf_ckpt_dir", type=str, default=None)
    parser.add_argument("--model_hf_path", type=str, default=None)
    parser.add_argument("--model_id_for_saving", type=str, required=True)
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--num_generations", type=int, default=1)
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    
    # inference
    parser.add_argument("--batch_size", type=int, default=500)
    
    args = parser.parse_args()
    
    #! Sanity check args
    assert "/" not in args.test_dataset_name
    assert "/" not in args.model_id_for_saving
    if not os.path.exists(args.prompt_config_file):
        pprint(f"Prompt config file not found at {args.prompt_config_file}!")
        exit()
        
    if args.chat_template_file is not None and not args.apply_chat_template:
        pprint(f"WARNING: chat template file provided but chat template is not being applied.")
        
    if args.num_generations > 1 and args.temperature == 0.0:
        pprint(f"num_generations > 1 but temperature is 0.0.")
        exit()
    
    assert not args.out_root.endswith("/")
    if args.seed != 42:
        args.out_root = args.out_root + f"-seed{args.seed}"
    
    pprint(f"Examinee args:\n{'-' * 42}\n{json.dumps(vars(args), indent=4)}\n{'-' * 42}")
    
    fix_seed(args.seed)
    main(args)
