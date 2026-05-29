#!/bin/bash
# Eagle 3.1 MLA Training Script
# This script converts a HuggingFace MLA base model (e.g. Kimi-K2, DeepSeek-V3) to
# Megatron format with Eagle 3.1 speculative decoding head, then finetunes it.
#
# Required environment variables:
#   HF_MODEL_CKPT: Path/name of the HuggingFace base model (e.g. "baseten/lovable-iterative-v5-refined-only-kimi-merged-deepseek")
#   EAGLE_CKPT (optional): Path to pretrained eagle weights to load (e.g. "lightseekorg/kimi-k2.6-eagle3.1-mla")
#
# The eagle architecture config (num_layers, fc_norm, norm_output, aux layer ids) is in
# conf/eagle31_mla_config.json. Base model architecture params (kv_lora_rank, q_lora_rank,
# num_attention_heads, etc.) are automatically derived from the base model during conversion.

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"

# Common arguments and base model specific arguments
source "${SCRIPT_DIR}/conf/arguments.sh"

# Set up cache dir for HF to avoid out of space error
export HF_DATASETS_CACHE="/tmp/hf_datasets_cache"

# Extra arguments of this script
MLM_DEFAULT_ARGS=" \
    --distributed-timeout-minutes 30 \
    --auto-detect-ckpt-format \
    --export-te-mcore-model \
    --finetune \
"

# Use eagle3-kimik2 algorithm for MLA models
EAGLE3_CONVERT_ARGS=" \
    --algorithm eagle3-kimik2 \
    --eagle-config ${SCRIPT_DIR}/conf/eagle31_mla_config.json \
    --export-offline-model \
"

# Eagle 3.1 specific args for finetune
EAGLE31_ARGS=" \
    --eagle-decoder-type kimik2 \
    --eagle-ttt-steps 3 \
    --eagle-loss-decay-factor 0.9 \
    --eagle-self-logit-distillation \
    --eagle-config-json ${SCRIPT_DIR}/conf/eagle31_mla_config.json \
"

if [ -z ${MLM_MODEL_SAVE} ]; then
    MLM_MODEL_SAVE=${MLM_WORK_DIR}/${MLM_MODEL_CFG}-Eagle31-MLA
    printf "${MLM_WARNING} Variable ${PURPLE}MLM_MODEL_SAVE${WHITE} is not set (default: ${MLM_MODEL_SAVE})!\n"
fi

if [ -z ${MLM_DATA_ARGS} ]; then
    MLM_DATA_ARGS=" \
        --train-samples 128000 \
        --lr-decay-samples 128000 \
        --lr-warmup-samples 0 \
        --split 100,0,0 \
        --finetune-hf-dataset Magpie-Align/Magpie-Llama-3.1-Pro-MT-300K-Filtered \
    "
fi

if [ -z ${MLM_TRAIN_ARGS} ]; then
    MLM_TRAIN_ARGS=" \
        --no-gradient-accumulation-fusion \
        --reset-position-ids \
        --reset-attention-mask \
        --eod-mask-loss \
        --micro-batch-size 1 \
        --attention-dropout 0.0 \
        --hidden-dropout 0.0 \
        --no-check-for-nan-in-loss-and-grad \
    "
fi

if [ -z ${MLM_OPTIM_ARGS} ]; then
    MLM_OPTIM_ARGS=" \
        --lr 5.0e-5 \
        --min-lr 1.0e-7 \
        --lr-decay-style cosine \
        --clip-grad 1.0 \
        --weight-decay 0.0 \
        --adam-beta1 0.9 \
        --adam-beta2 0.95 \
        --init-method-std 0.010 \
    "
fi

if [ -z ${MLM_EVAL_ARGS} ]; then
    MLM_EVAL_ARGS=" \
        --eval-iters 1 \
        --eval-interval 1000 \
        --save-interval 1000 \
        --log-interval 100 \
    "
fi

# Convert HF checkpoint to Megatron EAGLE3.1 MLA model if not exist
if [[ ! -d ${MLM_MODEL_SAVE} ]]; then
    ${LAUNCH_SCRIPT} ${SCRIPT_DIR}/convert_model.py \
        ${MODEL_ARGS} \
        --tensor-model-parallel-size ${TP} \
        --expert-tensor-parallel-size ${ETP} \
        --pipeline-model-parallel-size ${PP} \
        --expert-model-parallel-size ${EP} \
        --tokenizer-model ${TOKENIZER_MODEL} \
        --pretrained-model-path ${HF_MODEL_CKPT} \
        --save ${MLM_MODEL_SAVE} \
        ${MLM_DEFAULT_ARGS} ${EAGLE3_CONVERT_ARGS}
fi


${LAUNCH_SCRIPT} ${SCRIPT_DIR}/finetune.py \
    ${MODEL_ARGS} \
    --tensor-model-parallel-size ${TP} \
    --expert-tensor-parallel-size ${ETP} \
    --expert-model-parallel-size ${EP} \
    --pipeline-model-parallel-size ${PP} \
    --tokenizer-model ${TOKENIZER_MODEL} \
    --load ${MLM_MODEL_SAVE} \
    --save ${MLM_MODEL_SAVE} \
    ${MLM_DATA_ARGS} \
    ${MLM_OPTIM_ARGS} \
    ${MLM_TRAIN_ARGS} \
    ${MLM_EVAL_ARGS} \
    ${MLM_RESUME_ARGS} \
    ${MLM_DEFAULT_ARGS} ${EAGLE31_ARGS} ${MLM_EXTRA_ARGS}
