import logging
import torch
from evaluate import load
from transformers import AutoTokenizer, AutoModelForCausalLM
from common.scripts.get_models import ALL_CPTS, get_model_paths
from pretrain.fineweb_data import load_fineweb_validation_sample

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def calculate_loss_perplexity(model_path, num_samples=1000):
    """Calculate validation loss and perplexity using fineweb data."""
    logger.info(f"Evaluating model: {model_path}")
    
    logger.info("loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")
    model.eval()
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    logger.info("loading fineweb validation data...")
    data = load_fineweb_validation_sample(num_samples)
    
    predictions = [x["text"] for x in data]
    logger.info(f"loaded {len(predictions)} text samples")
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for text in predictions:
            if len(text.strip()) == 0:
                continue
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            outputs = model(**inputs, labels=inputs["input_ids"])
            total_loss += outputs.loss.item()
            num_batches += 1
    
    avg_val_loss = total_loss / num_batches if num_batches > 0 else float('inf')
    
    logger.info("calculating perplexity...")
    perplexity = load("perplexity", module_type="metric")
    results = perplexity.compute(model_id=model_path, predictions=predictions)
    
    logger.info(f"validation loss: {avg_val_loss:.4f} | mean perplexity: {results['mean_perplexity']:.4f}")
    
    return {
        "validation_loss": avg_val_loss,
        "mean_perplexity": results['mean_perplexity'],
        "perplexities": results['perplexities']
    }

def main():
    logger.info("starting pretraining validation loss + perplexity evaluations")

    model_size = ["0.5B", "1B", "4B"]
    pretraining = [1, 20, 40, 80, 160, 320]
    cpt = ALL_CPTS
    # sft = ([1, 2, 4, 8, 16, 32], [100, 200, 300, 400])
    # rl = ([1, 2, 4, 8, 16, 32], [100, 200, 300, 400])

    logger.info(f"Running pretraining evals for {model_size} | {pretraining} | {cpt}")

    # Number of validation samples to use from fineweb
    num_samples = 1000  # Adjust this number as needed

    results_all = []
    for model_path in get_model_paths(model_size, pretraining, cpt, SFTs=[None], RLs=[None]):
        results = calculate_loss_perplexity(model_path, num_samples)
        results["model_path"] = model_path
        results_all.append(results)
        
    logger.info(f"Completed evaluation of {len(results_all)} models")
    return results_all

if __name__ == "__main__":
    main() 