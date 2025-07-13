from datasets import load_dataset
import json

def generate_unused_gsm8k():

    print("loading used dataset")
    gsm8k_used = load_dataset("ZhentingNLP/mathaug", split="gsm8k_sampled_250k")
    print("loading full dataset")
    gsm8k_full = load_dataset("ZhentingNLP/mathaug", split="gsm8k")

    print("creating filter")
    used_ids = set(example['unique_id'] for example in gsm8k_used)
    gsm8k_unused = gsm8k_full.filter(lambda x: x['unique_id'] not in used_ids)

    data = gsm8k_unused.to_list()
    output_path = "/Users/kaitlynwang/evolm/finetune/llama-factory/sft_data/gsm8k_unused_sft.json"
    print(f"saved to output path {output_path}")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f)

if __name__ == "__main__":
    generate_unused_gsm8k()


