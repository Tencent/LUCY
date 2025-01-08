#!/bin/bash
WORK_DIR=$(pwd)

# ME=$(basename "$0")
# ME=${ME%.*}
# TIMESTAMP=$(date '+%m%d%y-%H%M%S')

CACHE_DIR=/mnt/data/hetinggao/models
MANIFEST_DIR="/mnt/data/hetinggao/manifest/LibriSpeech/sub10"
# CKPT_PATH=/mnt/data/hetinggao/vita-e2e/outputs/vita_qwen2_s1_en_parallel_tte-091824-112332/checkpoint-26000
# OUTPUT_PATH=/mnt/data/hetinggao/vita-e2e/generated/en_tte_26k
CKPT_PATH=/mnt/data/hetinggao/vita-e2e/outputs/vita_qwen2_s1_en_parallel_tte-091824-112332/checkpoint-40000
OUTPUT_PATH=/mnt/data/hetinggao/vita-e2e/generated/en_tte_40k
# MANIFEST_DIR="/mnt/data/hetinggao/manifest/WenetSpeech/sub10"
#OUTPUT_DIR=outputs/vita_qwen2_s1_parallel-091324-121440
#CKPT_PATH=$(ls ${OUTPUT_DIR}/checkpoint-* -atd | head -n1)
#CKPT_PATH=/mnt/data/hetinggao/vita-e2e/backups/vita_qwen2_s1_zh_parallel-091524-120824/checkpoint-50000
# CKPT_PATH=/mnt/data/hetinggao/vita-e2e/backups/vita_qwen2_s1_zh_parallel-091524-120824/checkpoint-100000
# CKPT_PATH=/mnt/data/hetinggao/vita-e2e/backups/vita_qwen2_s1_zh_parallel_tte-091724-063343/checkpoint-50000
# OUTPUT_PATH=/mnt/data/hetinggao/vita-e2e/generated/zh_tte_50k

# MANIFEST_DIR="/mnt/data/hetinggao/manifest/aishell3/sub10"
# CKPT_PATH=/mnt/data/hetinggao/vita-e2e/outputs/vita_qwen2_s1_zh_parallel_tte_aishell3-091824-174529/checkpoint-24000
# OUTPUT_PATH=/mnt/data/hetinggao/vita-e2e/generated/zh_tte_24k_aishell3
mkdir -p $OUTPUT_PATH

AUDIO_MANIFEST="${MANIFEST_DIR}/train.tsv"
TRANS_FILE="${MANIFEST_DIR}/train.wrd"
CODEC_FILE="${MANIFEST_DIR}/train.snac"

MODEL_NAME_OR_PATH="Qwen/Qwen2-1.5B"
AUDIO_ENCODER="openai/whisper-medium"

export PYTHONPATH=$WORK_DIR
python vita/scripts/infer_s1.py \
    --audio_manifest ${AUDIO_MANIFEST} \
    --transcript_file ${TRANS_FILE} \
    --codec_file ${CODEC_FILE} \
    --audio_feature_rate 50 \
    --sample_rate 16000 \
    --tasks "ASR" \
    --shuffle False \
    --model_type "qwen2" \
    --model_name_or_path /mnt/data/hetinggao/models/Qwen2-1.5B \
    --audio_encoder /mnt/data/hetinggao/models/whisper-medium \
    --model_hidden_size 1536 \
    --freeze_backbone True \
    --freeze_audio_encoder True \
    --audio_encoder_hidden_size 1024 \
    --audio_projector_hidden_size 7168 \
    --audio_num_codebook 7 \
    --text_vocab_size 151936 \
    --text_special_tokens 64 \
    --audio_vocab_size 4096 \
    --audio_special_tokens 64 \
    --cache_dir ${CACHE_DIR} \
    --text_additional "EOT" "PAD_T" "BOT" "ANS_T" "TTS" \
    --audio_additional "EOA" "PAD_A" "BOA" "ANS_A" "ASR" \
	--max_code_length 360 \
    --ckpt_path ${CKPT_PATH} \
    --output_path ${OUTPUT_PATH} \
    
    

    
    
    # --per_device_train_batch_size 32 \
    # --num_train_epochs 50 \
    # --save_strategy "steps" \
    # --save_steps 1000 \
    # --save_total_limit 10 \
    # --eval_strategy "no" \
    # --learning_rate 5e-4 \
    # --weight_decay 0. \
    # --warmup_ratio 0.03 \
    # --logging_steps 50 \
    # --lr_scheduler_type "cosine" \
    # --gradient_checkpointing True \
    # --bf16 True \

    
    
    
    # --output_dir ${OUPTUT_DIR} \
    
    # --dataloader_num_workers 4 \
    # --remove_unused_columns False \
    
