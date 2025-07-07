from typing import Literal, Dict, Any, TypedDict, Tuple, List
from huggingface_hub import repo_exists
from itertools import product


MODEL_SIZE = Literal["0.5B", "1B", "4B"]
PRETRAINING = Literal[10, 20, 40, 80, 160, 320]
CPT = Literal[None, "FM10", "FM20", "FM30", "FM40", "FM50", "FW8FM12", "FW8FM22", "FW8FM32", "FW8FM42", "FW1_6FM48_4", "FW16FM34"]
SFT = Tuple[Literal[1, 2, 4, 8, 16, 32], Literal[100, 200, 300, 400]]  
RL = Tuple[Literal[1, 2, 4, 8, 16, 32], Literal[100, 200, 300, 400]] 

ALL_CPTS = ["", "FM10", "FM20", "FM30", "FM40", "FM50", "FW8FM12", "FW8FM22", "FW8FM32", "FW8FM42", "FW1_6FM48_4", "FW16FM34"]

def check_hf_model_exists(model_path: str) -> bool:
    try:
        return repo_exists(model_path)
    except Exception:
        print(f"Tried HF model path {model_path} but didn't exist")
        return False


def get_model_paths(model_sizes: List[MODEL_SIZE], pretrainings: List[PRETRAINING], CPTs: List[CPT] = None, SFTs: List[SFT] = None, RLs: List[RL] = None) -> List[str]:
    paths = []
    
    for model_size, pretraining, cpt, sft, rl in product(model_sizes, pretrainings, CPTs, SFTs, RLs): # generate all possible combos
        path_parts = [f"zhenting/myllama-{model_size}", f"{pretraining}BT"]
        
        if cpt is not None and cpt != '':
                path_parts.append("cpt")
                path_parts.append(f"Mixed{cpt}")
            
        if sft is not None:
            path_parts.append(f"sftep{sft[0]}")
            path_parts.append(f"sampled{sft[1]}k_first100k_qwen7b")
            
        if rl is not None:
            path_parts.append(f"rlep{rl[0]}")
            path_parts.append(f"last{rl[1]}k")

        path = "-".join(path_parts)

        if check_hf_model_exists(path):
            paths.append(path)
    
    return paths



