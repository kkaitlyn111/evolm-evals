# Simple Modular Evaluation

A simple, modular evaluation script for zero-shot accuracy and log likelihood evaluation on mathematical reasoning datasets. Designed for easy batching with Slurm across multiple GPUs.

## Features

- **Zero-shot accuracy evaluation**: Tests model performance without few-shot examples
- **Top-k accuracy evaluation**: Generates multiple responses with increasing temperature and computes top-k success rates (k=1 to 10)
- **Log likelihood computation**: Calculates log likelihood of ground truth CoT + answer using HuggingFace forward pass
- **Modular design**: Evaluates single model + single dataset per run for easy batching
- **Multiple datasets**: Supports all mathematical reasoning datasets in the codebase
- **HuggingFace only**: No VLLM dependency, pure HuggingFace implementation

## Supported Datasets

- `AIME2023`, `AIME2024` - American Invitational Mathematics Examination
- `GSM8KPlatinum` - Grade School Math 8K (Premium version) 
- `TinyGSM` - Smaller version of GSM8K
- `MATH` - Competition mathematics problems
- `MATH500`, `MATHHard` - MATH dataset variants
- `MATHLevel1` through `MATHLevel4` - MATH by difficulty level
- `MinervaMath` - Mathematical reasoning dataset
- `OlympiadBench` - Mathematical olympiad problems

## Installation

```bash
cd evaluation
pip install -r requirements.txt
```

## Usage

### Single Evaluation

Evaluate one model on one dataset with top-10 accuracy:

```bash
python evaluation/simple_eval.py \
    --model "meta-llama/Llama-3.1-8B-Instruct" \
    --dataset "MATH" \
    --max_new_tokens 1024 \
    --k 10 \
    --device cuda \
    --output_dir "evaluation/results"
```

### Quick Test

Test on a small subset with top-5 accuracy:

```bash
python evaluation/simple_eval.py \
    --model "microsoft/DialoGPT-medium" \
    --dataset "TinyGSM" \
    --max_examples 10 \
    --max_new_tokens 256 \
    --k 5 \
    --device cuda
```

### Batch Evaluation with Slurm

1. **Edit the batch script**: Modify `evaluation/batch_eval_example.sh`:
   - Update the project path
   - Adjust models and datasets lists
   - Set appropriate Slurm parameters for your cluster

2. **Submit the batch job**:
   ```bash
   sbatch evaluation/batch_eval_example.sh
   ```

3. **Monitor progress**:
   ```bash
   squeue -u $USER
   tail -f logs/eval_*.out
   ```

## Output

The script generates two files per evaluation:

1. **Summary**: `{model}_{dataset}_summary_{timestamp}.json`
   - Overall metrics: top-k accuracies (k=1 to 10), log likelihood, perplexity
   - Counts and rates

2. **Detailed Results**: `{model}_{dataset}_results_{timestamp}.jsonl`
   - Per-example results with all k responses, predictions, ground truth, and log likelihoods

## Example Output

```
EVALUATION SUMMARY
======================================================================
Model: meta-llama/Llama-3.1-8B-Instruct
Dataset: MATH
Total examples: 500
Valid prediction rate: 0.9600

TOP-K ACCURACIES:
  Top-1: 0.4680
  Top-2: 0.5240
  Top-3: 0.5680
  Top-4: 0.5980
  Top-5: 0.6220
  Top-6: 0.6420
  Top-7: 0.6580
  Top-8: 0.6720
  Top-9: 0.6840
  Top-10: 0.6940

Average log likelihood: -2.3456
Perplexity: 10.43
Evaluation time: 425.89 seconds
======================================================================
```

## How Top-K Evaluation Works

The script generates k responses per problem with increasing temperature:
- **Response 1**: Temperature 0.0 (greedy decoding)
- **Response 2**: Temperature 0.1 
- **Response 3**: Temperature 0.2
- **...**
- **Response k**: Temperature (k-1)*0.1

For each k from 1 to 10, **top-k accuracy** measures whether ANY of the first k responses is correct. This gives insight into model consistency and the benefit of multiple attempts.

## Performance Notes

- **Memory**: Models load with `device_map="auto"` for automatic GPU placement
- **Speed**: ~0.1-0.2 examples/second for 8B models, ~0.03-0.05 examples/second for 70B models (with k=10)
- **Time estimates** (with k=10): 
  - GSM8K (1.3K examples) with 8B model: ~2-3 hours
  - MATH (5K examples) with 70B model: ~24-30 hours
  - AIME (30 examples) with any model: ~5-10 minutes
- **Scaling**: Evaluation time scales approximately linearly with k

## Customization

The script is designed to be simple but extensible:

- **Add new datasets**: Add entries to `DATASET_EVALUATOR_MAP` 
- **Change prompts**: Modify `format_problem_prompt()`
- **Adjust generation**: Update parameters in `generate_response()` or `generate_multiple_responses()`
- **Custom temperature schedule**: Modify the temperature calculation in `generate_multiple_responses()`
- **Different k values**: Use `--k` parameter (1-10) to control number of responses generated
- **Custom evaluators**: Use existing evaluator classes from the codebase

## Troubleshooting

- **OOM errors**: Reduce `max_new_tokens`, use smaller models, or reduce `--k`
- **Very slow evaluation**: Use `--max_examples` for testing, or reduce `--k` for faster runs
- **Wrong evaluator**: Check dataset name matches exactly (case-sensitive)
- **Missing datasets**: Ensure JSONL files exist in `evaluation/cot-eval-harness/data/cooked/`
- **Invalid k parameter**: Ensure `--k` is between 1 and 10