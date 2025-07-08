#!/usr/bin/env python3
"""
Simple modular evaluation script for zero-shot accuracy and log likelihood evaluation.
Evaluates a single model on a single dataset for easy batching with Slurm.
Supports all mathematical reasoning datasets in the codebase.
"""

import os
import json
import torch
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser
from typing import List, Dict, Optional, Tuple
import sys
import time
from pathlib import Path

# Add the cot-eval-harness to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'cot-eval-harness'))

from cot_eval.evaluation.Evaluator import get_evaluator
from transformers import AutoTokenizer, AutoModelForCausalLM
import warnings
warnings.filterwarnings("ignore")


# Dataset to evaluator mapping based on the codebase
DATASET_EVALUATOR_MAP = {
    'AIME2023': 'AIME',
    'AIME2024': 'AIME', 
    'GSM8KPlatinum': 'GSM8K',
    'TinyGSM': 'GSM8K',
    'MATH': 'MATH',
    'MATH500': 'MATH',
    'MATHHard': 'MATH',
    'MATHLevel1': 'MATH',
    'MATHLevel2': 'MATH', 
    'MATHLevel3': 'MATH',
    'MATHLevel4': 'MATH',
    'MinervaMath': 'MATH',
    'OlympiadBench': 'MATH'
}

SUPPORTED_DATASETS = list(DATASET_EVALUATOR_MAP.keys())


def load_dataset(dataset_name: str) -> List[Dict]:
    """Load a dataset from the cooked data directory."""
    data_path = f"evaluation/cot-eval-harness/data/cooked/{dataset_name}.jsonl"
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset {dataset_name} not found at {data_path}")
    
    data = []
    with open(data_path, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    
    print(f"Loaded {len(data)} examples from {dataset_name}")
    return data


def load_model_and_tokenizer(model_name: str, device: str = "cuda"):
    """Load HuggingFace model and tokenizer."""
    print(f"Loading model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True
    )
    
    if device != "cuda":
        model = model.to(device)
    
    model.eval()
    print(f"Model loaded on device: {next(model.parameters()).device}")
    return model, tokenizer


def format_problem_prompt(problem: str) -> str:
    """Format the problem as a zero-shot prompt."""
    return f"Solve the following problem step by step.\n\nProblem: {problem}\n\nSolution:"


def generate_response(model, tokenizer, prompt: str, max_new_tokens: int = 512, temperature: float = 0.0) -> str:
    """Generate a single response from the model."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0.0,
            temperature=temperature if temperature > 0.0 else 1.0,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Extract only the generated part
    generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return response.strip()


def generate_multiple_responses(model, tokenizer, prompt: str, k: int = 10, max_new_tokens: int = 512) -> List[str]:
    """Generate k responses with increasing temperature for top-k evaluation."""
    responses = []
    
    # Generate responses with increasing temperature
    for i in range(k):
        # Temperature increases from 0.0 (greedy) to 1.0
        temperature = i * 0.1 if i > 0 else 0.0
        response = generate_response(model, tokenizer, prompt, max_new_tokens, temperature)
        responses.append(response)
    
    return responses


def compute_log_likelihood(model, tokenizer, prompt: str, target_text: str) -> float:
    """Compute log likelihood of target_text given the prompt."""
    full_text = prompt + target_text
    
    # Tokenize the full text
    inputs = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Tokenize just the prompt to know where target starts
    prompt_inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    prompt_length = prompt_inputs['input_ids'].shape[1]
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    
    # Get log probabilities
    log_probs = torch.log_softmax(logits, dim=-1)
    
    # Extract log probabilities for the target tokens
    target_tokens = inputs['input_ids'][0][prompt_length:]
    target_log_probs = []
    
    for i, token_id in enumerate(target_tokens):
        if prompt_length + i < log_probs.shape[1]:
            token_log_prob = log_probs[0, prompt_length + i - 1, token_id].item()
            target_log_probs.append(token_log_prob)
    
    if len(target_log_probs) == 0:
        return float('-inf')
    
    # Return average log likelihood per token
    return np.mean(target_log_probs)


def evaluate_single_example(model, tokenizer, evaluator, example: Dict, max_new_tokens: int = 512, k: int = 10) -> Dict:
    """Evaluate a single example for both accuracy, top-k accuracy, and log likelihood."""
    problem = example['problem']
    prompt = format_problem_prompt(problem)
    
    # Generate multiple responses for top-k evaluation
    responses = generate_multiple_responses(model, tokenizer, prompt, k, max_new_tokens)
    
    # Extract ground truth answer
    if 'gt_answer' in example and example['gt_answer'] is not None:
        gt_answer = str(example['gt_answer'])
    elif 'gt_solution' in example:
        gt_answer = evaluator.extract_answer_from_gold_solution(example['gt_solution'])
    else:
        gt_answer = None
    
    # Evaluate each response
    predicted_answers = []
    is_correct_list = []
    
    for response in responses:
        predicted_answer = evaluator.extract_answer_from_model_completion(response)
        predicted_answers.append(predicted_answer)
        
        # Check if this response is correct
        is_correct = False
        if predicted_answer is not None and gt_answer is not None:
            is_correct = evaluator.check_answers_equiv(predicted_answer, gt_answer)
            if is_correct is None:  # Timeout or error in comparison
                is_correct = False
        is_correct_list.append(is_correct)
    
    # Compute top-k accuracy for k=1 to k=10
    top_k_correct = {}
    for top_k in range(1, k + 1):
        # Check if any of the top k responses are correct
        top_k_correct[f'top_{top_k}'] = any(is_correct_list[:top_k])
    
    # Compute log likelihood of ground truth solution (using greedy response prompt)
    gt_log_likelihood = float('-inf')
    if 'gt_solution' in example and example['gt_solution']:
        try:
            gt_log_likelihood = compute_log_likelihood(model, tokenizer, prompt, example['gt_solution'])
        except Exception as e:
            print(f"Error computing log likelihood: {e}")
            gt_log_likelihood = float('-inf')
    
    return {
        'id': example.get('id', 'unknown'),
        'problem': problem,
        'prompt': prompt,
        'generated_responses': responses,
        'predicted_answers': predicted_answers,
        'gt_answer': gt_answer,
        'gt_solution': example.get('gt_solution', ''),
        'is_correct_list': is_correct_list,
        'top_k_correct': top_k_correct,
        'gt_log_likelihood': gt_log_likelihood
    }


def evaluate_model_on_dataset(
    model_name: str,
    dataset_name: str,
    max_examples: Optional[int] = None,
    max_new_tokens: int = 512,
    k: int = 10,
    device: str = "cuda"
) -> Dict:
    """Evaluate a model on a dataset."""
    
    if dataset_name not in SUPPORTED_DATASETS:
        raise ValueError(f"Dataset {dataset_name} not supported. Supported: {SUPPORTED_DATASETS}")
    
    # Load data
    data = load_dataset(dataset_name)
    if max_examples:
        data = data[:max_examples]
    
    # Load model
    model, tokenizer = load_model_and_tokenizer(model_name, device)
    
    # Get evaluator
    evaluator_type = DATASET_EVALUATOR_MAP[dataset_name]
    evaluator = get_evaluator(evaluator_type, answer_extraction_format="both")
    
    # Initialize metrics tracking
    results = []
    log_likelihoods = []
    top_k_counts = {f'top_{i}': 0 for i in range(1, k + 1)}
    
    print(f"Evaluating {len(data)} examples with top-{k} accuracy...")
    for i, example in enumerate(tqdm(data)):
        try:
            result = evaluate_single_example(model, tokenizer, evaluator, example, max_new_tokens, k)
            results.append(result)
            
            # Track top-k metrics
            for top_k_key, is_correct in result['top_k_correct'].items():
                if is_correct:
                    top_k_counts[top_k_key] += 1
            
            # Track log likelihood
            if result['gt_log_likelihood'] != float('-inf'):
                log_likelihoods.append(result['gt_log_likelihood'])
                
        except Exception as e:
            print(f"Error processing example {i}: {e}")
            # Add a failed result
            failed_top_k = {f'top_{i}': False for i in range(1, k + 1)}
            result = {
                'id': example.get('id', f'example_{i}'),
                'problem': example.get('problem', ''),
                'error': str(e),
                'is_correct_list': [False] * k,
                'top_k_correct': failed_top_k,
                'predicted_answers': [None] * k,
                'gt_answer': None,
                'gt_solution': example.get('gt_solution', ''),
                'gt_log_likelihood': float('-inf')
            }
            results.append(result)
    
    # Compute metrics
    total_examples = len(results)
    avg_log_likelihood = np.mean(log_likelihoods) if log_likelihoods else float('-inf')
    perplexity = np.exp(-avg_log_likelihood) if avg_log_likelihood != float('-inf') else float('inf')
    
    # Compute top-k accuracies
    top_k_accuracies = {}
    for top_k_key, count in top_k_counts.items():
        top_k_accuracies[f'{top_k_key}_accuracy'] = count / total_examples if total_examples > 0 else 0.0
    
    # Count valid predictions (for top-1)
    valid_predictions = sum(1 for r in results if 'predicted_answers' in r and r['predicted_answers'][0] is not None)
    valid_prediction_rate = valid_predictions / total_examples if total_examples > 0 else 0.0
    
    summary = {
        'model_name': model_name,
        'dataset_name': dataset_name,
        'total_examples': total_examples,
        'k': k,
        'valid_predictions': valid_predictions,
        'valid_prediction_rate': valid_prediction_rate,
        'avg_log_likelihood': avg_log_likelihood,
        'perplexity': perplexity,
        'log_likelihood_count': len(log_likelihoods),
        **top_k_accuracies
    }
    
    return {
        'summary': summary,
        'results': results
    }


def save_results(evaluation_results: Dict, output_dir: str):
    """Save evaluation results to files."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    model_name = evaluation_results['summary']['model_name'].replace('/', '_')
    dataset_name = evaluation_results['summary']['dataset_name']
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # Save summary
    summary_file = f"{output_dir}/{model_name}_{dataset_name}_summary_{timestamp}.json"
    with open(summary_file, 'w') as f:
        json.dump(evaluation_results['summary'], f, indent=2)
    
    # Save detailed results
    results_file = f"{output_dir}/{model_name}_{dataset_name}_results_{timestamp}.jsonl"
    with open(results_file, 'w') as f:
        for result in evaluation_results['results']:
            f.write(json.dumps(result) + '\n')
    
    print(f"Results saved to {output_dir}")
    print(f"Summary: {summary_file}")
    print(f"Detailed results: {results_file}")


def main():
    parser = ArgumentParser(description="Simple modular evaluation for mathematical reasoning with top-k accuracy")
    parser.add_argument("--model", type=str, required=True, help="HuggingFace model name")
    parser.add_argument("--dataset", type=str, required=True, choices=SUPPORTED_DATASETS, 
                       help=f"Dataset name. Supported: {SUPPORTED_DATASETS}")
    parser.add_argument("--max_examples", type=int, default=None, help="Maximum number of examples to evaluate")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Maximum new tokens to generate")
    parser.add_argument("--k", type=int, default=10, help="Number of responses to generate for top-k accuracy (1-10)")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda, cpu)")
    parser.add_argument("--output_dir", type=str, default="evaluation/results", help="Output directory for results")
    
    args = parser.parse_args()
    
    # Validate k parameter
    if args.k < 1 or args.k > 10:
        raise ValueError("k must be between 1 and 10")
    
    print(f"Starting evaluation:")
    print(f"  Model: {args.model}")
    print(f"  Dataset: {args.dataset}")
    print(f"  Max examples: {args.max_examples or 'All'}")
    print(f"  Top-k evaluation: k={args.k}")
    print(f"  Device: {args.device}")
    print()
    
    # Run evaluation
    start_time = time.time()
    results = evaluate_model_on_dataset(
        model_name=args.model,
        dataset_name=args.dataset,
        max_examples=args.max_examples,
        max_new_tokens=args.max_new_tokens,
        k=args.k,
        device=args.device
    )
    end_time = time.time()
    
    # Print summary
    summary = results['summary']
    print("\n" + "="*70)
    print("EVALUATION SUMMARY")
    print("="*70)
    print(f"Model: {summary['model_name']}")
    print(f"Dataset: {summary['dataset_name']}")
    print(f"Total examples: {summary['total_examples']}")
    print(f"Valid prediction rate: {summary['valid_prediction_rate']:.4f}")
    print()
    
    # Print top-k accuracies
    print("TOP-K ACCURACIES:")
    for k_val in range(1, args.k + 1):
        acc_key = f'top_{k_val}_accuracy'
        if acc_key in summary:
            print(f"  Top-{k_val}: {summary[acc_key]:.4f}")
    print()
    
    print(f"Average log likelihood: {summary['avg_log_likelihood']:.4f}")
    print(f"Perplexity: {summary['perplexity']:.4f}")
    print(f"Evaluation time: {end_time - start_time:.2f} seconds")
    print("="*70)
    
    # Save results
    save_results(results, args.output_dir)


if __name__ == "__main__":
    main() 