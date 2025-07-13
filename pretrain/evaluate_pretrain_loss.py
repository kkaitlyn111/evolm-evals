import sys
import os
# Add the evolm-evals root directory to Python path so we can import common module
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)  # Go up from pretrain/ to evolm-evals/
sys.path.insert(0, project_root)

import logging
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from common.scripts.get_models import get_model_paths
from pretrain.fineweb_data import load_fineweb_validation_sample
from itertools import product

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def calculate_perplexity_manual(model, tokenizer, texts, batch_size=5):
    """Manually calculate perplexity using the loaded model."""
    model.eval()
    all_perplexities = []
    
    total_batches = (len(texts) + batch_size - 1) // batch_size
    logger.info(f"Processing {len(texts)} texts in {total_batches} batches of {batch_size}")
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Calculating Perplexity", unit="batch"):
        batch_texts = texts[i:i+batch_size]
        
        # Clear GPU cache before each batch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        batch_perplexities = []
        
        for text in batch_texts:
            # Tokenize the text
            inputs = tokenizer(text, return_tensors="pt")
            input_ids = inputs.input_ids.to(model.device)
            
            # Calculate loss
            with torch.no_grad():
                outputs = model(input_ids, labels=input_ids)
                loss = outputs.loss
                perplexity = torch.exp(loss).item()
                batch_perplexities.append(perplexity)
        
        all_perplexities.extend(batch_perplexities)
        
        batch_num = i//batch_size + 1
        batch_mean = sum(batch_perplexities) / len(batch_perplexities)
        logger.info(f"  Batch {batch_num}/{total_batches}: mean perplexity = {batch_mean:.4f}")
        
        # Clear GPU cache after each batch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # Log memory usage every 10 batches
        if batch_num % 10 == 0:
            logger.info(f"    GPU Memory: {torch.cuda.memory_allocated()/1024**3:.2f}GB allocated")
    
    return all_perplexities


def calculate_perplexity(model_path, num_samples=1000):
    """Calculate perplexity using fineweb data."""
    logger.info(f"=" * 60)
    logger.info(f"Evaluating model: {model_path}")
    logger.info(f"=" * 60)
    
    logger.info("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")
    model.eval()
    logger.info(f"✓ Model loaded successfully")
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    logger.info(f"Loading {num_samples} fineweb validation samples...")
    data = load_fineweb_validation_sample(num_samples)
    
    predictions = [x["text"] for x in data]
    logger.info(f"✓ Loaded {len(predictions)} text samples")
    
    logger.info("Starting perplexity calculation...")
    logger.info(f"GPU Memory before starting: {torch.cuda.memory_allocated()/1024**3:.2f}GB allocated, {torch.cuda.memory_reserved()/1024**3:.2f}GB reserved")
    
    # Use manual perplexity calculation to avoid loading model twice
    batch_size = 2  # Even smaller batch size since we're processing one text at a time
    all_perplexities = calculate_perplexity_manual(model, tokenizer, predictions, batch_size)
    
    mean_perplexity = sum(all_perplexities) / len(all_perplexities) if all_perplexities else float('inf')
    
    logger.info(f"✓ FINAL RESULT - Mean perplexity: {mean_perplexity:.4f}")
    logger.info(f"  Perplexity range: [{min(all_perplexities):.4f}, {max(all_perplexities):.4f}]")
    
    return {
        "mean_perplexity": mean_perplexity,
        "perplexities": all_perplexities
    }

def main():
    logger.info("Starting pretraining perplexity evaluations")

    # model_size = ["0.5B", "1B", "4B"]
    # pretraining = [10, 20, 40, 80, 160, 320]
    # cpt = ["", "FM10BT", "FM20BT", "FM30BT", "FM40BT", "MixedFW8FM12", "MixedFW8FM22", "MixedFW8FM32", "MixedFW8FM42", "MixedFW1_6FM48_4", "MixedFW16FM34"]
    # sft = [(None, None)] + list(product([1, 2, 4, 8, 16, 32], [100, 200, 300, 400]))
    # rl = [(None, None)] + list(product([1, 2, 4, 8, 16, 32], [100, 200, 300, 400]))
    model_size = ["1B"]
    pretraining = [20]
    cpt = ["MixedFW8FM42"]
    sft = [(1, 100)]
    rl = [(8, 100)]


    logger.info(f"Configuration:")
    logger.info(f"  Model sizes: {model_size}")
    logger.info(f"  Pretraining steps: {pretraining}")
    logger.info(f"  CPT variants: {cpt}")

    # Number of validation samples to use from fineweb
    num_samples = 500  # Reduced from 1000 to help with memory usage
    logger.info(f"  Validation samples: {num_samples}")

    # Get all model paths first to show total count
    model_paths = list(get_model_paths(model_size, pretraining, cpt, SFTs=sft, RLs=rl))
    print(model_paths)
    logger.info(f"\nTotal models to evaluate: {len(model_paths)}")
    
    results_all = []
    
    for idx, model_path in enumerate(tqdm(model_paths, desc="Evaluating Models", unit="model"), 1):
        logger.info(f"\nProgress: {idx}/{len(model_paths)} models")
        try:
            results = calculate_perplexity(model_path, num_samples)
            results["model_path"] = model_path
            results_all.append(results)
            logger.info(f"Model {idx} completed successfully")
        except Exception as e:
            logger.error(f"Model {idx} failed: {str(e)}")
            # Continue with next model instead of crashing
            continue
        
    logger.info(f"Successfully evaluated {len(results_all)}/{len(model_paths)} models")
    
    if results_all:
        perplexities = [r["mean_perplexity"] for r in results_all]
        logger.info(f"Overall perplexity stats:")
        logger.info(f"  Mean: {sum(perplexities)/len(perplexities):.4f}")
        logger.info(f"  Min:  {min(perplexities):.4f}")
        logger.info(f"  Max:  {max(perplexities):.4f}")
    
    return results_all

if __name__ == "__main__":
    main() 