"""
Simple utility to load fineweb validation data from HuggingFace.
"""

import datasets
import logging
import random

logger = logging.getLogger(__name__)

def load_fineweb_validation_sample(num_samples=1000, seed=42):
    """Load a sample of fineweb data for validation directly from HuggingFace."""
    logger.info(f"Loading {num_samples} samples from fineweb...")
    
    # Load fineweb dataset directly from HuggingFace
    dataset = datasets.load_dataset("HuggingFaceFW/fineweb", "sample-350BT", split="train", streaming=True)
    
    # Take a random sample
    samples = []
    random.seed(seed)
    
    for i, example in enumerate(dataset):
        if len(samples) >= num_samples:
            break
        
        # Sample every ~1000th example to get variety
        if random.random() < 0.001:  # 0.1% sampling rate
            samples.append({"text": example["text"]})
    
    # If we didn't get enough samples, take first num_samples
    if len(samples) < num_samples:
        logger.info(f"Only got {len(samples)} samples, taking first {num_samples} directly...")
        samples = []
        for i, example in enumerate(dataset):
            if i >= num_samples:
                break
            samples.append({"text": example["text"]})
    
    logger.info(f"Loaded {len(samples)} validation samples from fineweb")
    return samples

if __name__ == "__main__":
    # Test the data loader
    logging.basicConfig(level=logging.INFO)
    data = load_fineweb_validation_sample(10)
    print(f"Sample text: {data[0]['text'][:200]}...") 