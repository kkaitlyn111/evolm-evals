from typing import Literal, Dict, Any, TypedDict, Tuple, List
from huggingface_hub import repo_exists
from itertools import product


MODEL_SIZE = Literal["0.5B", "1B", "4B"]
PRETRAINING = Literal[10, 20, 40, 80, 160, 320]
CPT = Literal[None, "FM10BT", "FM20BT", "FM30BT", "FM40BT", "MixedFW8FM12", "MixedFW8FM22", "MixedFW8FM32", "MixedFW8FM42", "MixedFW1_6FM48_4", "MixedFW16FM34"]
SFT = Tuple[Literal[1, 2, 4, 8, 16, 32], Literal[100, 200, 300, 400]]  
RL = Tuple[Literal[1, 2, 4, 8, 16, 32], Literal[100, 200, 300, 400]] 

def check_hf_model_exists(model_path: str) -> bool:
    try:
        return repo_exists(model_path)
    except Exception:
        print(f"Tried HF model path {model_path} but didn't exist")
        return False


def get_model_paths(model_sizes: List[MODEL_SIZE], pretrainings: List[PRETRAINING], CPTs: List[CPT] = None, SFTs: List[SFT] = None, RLs: List[RL] = None) -> List[str]:
    paths = []
    seen_paths = set()
    
    for model_size, pretraining, cpt, sft, rl in product(model_sizes, pretrainings, CPTs, SFTs, RLs): # generate all possible combos
        path_parts = [f"zhenting/myllama-{model_size}", f"{pretraining}BT"]
        
        if cpt is not None and cpt != '' and model_size != "0.5B":
                path_parts.append("cpt")
                path_parts.append(f"{cpt}")
            
        if sft is not None and all(sft) and model_size != "0.5B":
            path_parts.append(f"sftep{sft[0]}")
            path_parts.append(f"sampled500k_first{sft[1]}k_qwen7b")
            
        if rl is not None and all(rl) and model_size != "0.5B":
            path_parts.append(f"rlep{rl[0]}")
            path_parts.append(f"last{rl[1]}k")

        path = "-".join(path_parts)
        
        if path in seen_paths:
            continue
        seen_paths.add(path)

        if check_hf_model_exists(path):
            paths.append(path)
            print("good path: ", path)
        else:
            print ("path not found: ", path)
    
    return paths



