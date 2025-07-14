OUT_ROOT=/juice5b/scr5b/kaitwang/evolm-evals/evaluation/cot-eval-harness/results

function collect_responses__greedy() {
    dataset_name=$1
    model_ckpt_dir=$2
    model_id_for_saving=$3
    prompt_config_file=$4
    python collect_responses.py \
        --test_dataset_name ${dataset_name} \
        --model_ckpt_dir ${model_ckpt_dir} \
        --model_hf_ckpt_dir ${model_ckpt_dir} \
        --model_hf_path ${model_ckpt_dir} \
        --model_id_for_saving ${model_id_for_saving} \
        --prompt_config_file ${prompt_config_file} \
        --repetition_penalty 1.1 \
        --temperature 0.0 \
        --num_generations 1 \
        --max_model_len 2048 \
        --out_root ${OUT_ROOT} \
        --force

}

export CUDA_VISIBLE_DEVICES=0
export VLLM_CONFIG_DIR=/juice5b/scr5b/kaitwang/.config/vllm
export VLLM_USAGE_STATS_DIR=/juice5b/scr5b/kaitwang/.config/vllm
export HF_HOME=/juice5b/scr5b/kaitwang/.cache/huggingface
export TRANSFORMERS_CACHE=/juice5b/scr5b/kaitwang/.cache/huggingface/transformers
export HF_DATASETS_CACHE=/juice5b/scr5b/kaitwang/.cache/huggingface/datasets
export HF_HUB_CACHE=/juice5b/scr5b/kaitwang/.cache/huggingface/hub
export TORCH_HOME=/juice5b/scr5b/kaitwang/.cache/torch
export XDG_CACHE_HOME=/juice5b/scr5b/kaitwang/.cache
export XDG_CONFIG_HOME=/juice5b/scr5b/kaitwang/.config
export XDG_DATA_HOME=/juice5b/scr5b/kaitwang/.local/share
export TMPDIR=/juice5b/scr5b/kaitwang/tmp
export TEMP=/juice5b/scr5b/kaitwang/tmp
export TMP=/juice5b/scr5b/kaitwang/tmp

model_list=(
    zhenting/myllama-1B-160BT-cpt-MixedFW8FM42
    # zhenting/myllama-1B-160BT-cpt-MixedFW8FM42-sftep1-sampled500k_first100k_qwen7b
    # zhenting/myllama-4B-160BT-cpt-MixedFW8FM42-sftep1-sampled500k_first100k_qwen7b-rlep1-last100k
    # zhenting/myllama-4B-160BT-cpt-MixedFW8FM42-sftep1-sampled500k_first100k_qwen7b-rlep16-last100k
    # zhenting/myllama-4B-160BT-cpt-MixedFW8FM42-sftep1-sampled500k_first100k_qwen7b-rlep2-last100k
)

for model_ckpt_dir in "${model_list[@]}";
do

    # dataset_name=GSM8KPlatinum,MATHLevel1,MATHLevel2,MATHLevel3,MATHLevel4,MATHHard,CRUXEval,BoardgameQA500,TabMWP,StrategyQA500
    dataset_name=GSM8KPlatinum
    prompt_config_file=prompts/myllama/default_dataset/prompt_config.json

    model_id_for_saving=$(basename "$model_ckpt_dir")--greedy
    collect_responses__greedy ${dataset_name} ${model_ckpt_dir} ${model_id_for_saving} ${prompt_config_file}

done