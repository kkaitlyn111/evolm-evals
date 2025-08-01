import os
import datasets
import argparse
import re


def extract_solution(solution_str):
    solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
    assert solution is not None
    final_solution = solution.group(0)
    final_solution = final_solution.split('#### ')[1].replace(',', '')
    return final_solution


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_src_dir', default='/path/to/rl')
    parser.add_argument('--dataset_name', default='gsm8k')
    args = parser.parse_args()
    
    local_dir = os.path.join(args.data_src_dir, args.dataset_name)
    os.makedirs(local_dir, exist_ok=True)
    
    # add a row to each data item that represents a unique id
    def make_map_fn(split):

        def process_fn(example, idx):
            problem = example.pop('question')
            gt_solution = example.pop('answer')
            gt_answer = extract_solution(gt_solution)
            data = {
                "data_source": "ZhentingNLP/mathaug",
                "prompt": [{
                    "role": "user",
                    "content": problem
                }],
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": gt_answer
                },
                "extra_info": {
                    'split': split,
                    'index': idx
                }
            }
            return data

        return process_fn

    data_source = 'openai/gsm8k'
    dataset = datasets.load_dataset(data_source, 'main', trust_remote_code=True)
    
    test_dataset = dataset['test']
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))
    