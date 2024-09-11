#!/bin/bash
WORK_DIR=$(pwd)

ME=$(basename "$0")
ME=${ME%.*}
TIMESTAMP=$(date '+%m%d%y-%H%M%S')

OUPTUT_DIR=${WORK_DIR}/outputs/${ME}-${TIMESTAMP}
# CACHE_DIR=/data/workspace/.cache
CACHE_DIR=/mnt/data/hetinggao/models
# mkdir -p $OUPTUT_DIR
MANIFEST_DIR="/mnt/data/hetinggao/manifest/sub100"
# MANIFEST_DIR="/data/workspace/manifest/sub100"

AUDIO_MANIFEST="${MANIFEST_DIR}/train.tsv"
TRANS_FILE="${MANIFEST_DIR}/train.wrd"
CODEC_FILE="${MANIFEST_DIR}/train.snac"

MODEL_NAME_OR_PATH="Qwen/Qwen2-1.5B"
AUDIO_ENCODER="openai/whisper-medium"

export PYTHONPATH=$WORK_DIR
python vita/scripts/train_s1.py \
    --model_type "qwen2" \
    --model_name_or_path /mnt/data/hetinggao/models/Qwen2-1.5B \
    --audio_encoder /mnt/data/hetinggao/models/whisper-medium \
    --freeze_audio_encoder True \
    --per_device_train_batch_size 16 \
    --num_train_epochs 5 \
    --save_strategy "steps" \
    --save_steps 10 \
    --save_total_limit 2 \
    --eval_strategy "no" \
    --learning_rate 5e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --logging_steps 10 \
    --lr_scheduler_type "cosine" \
    --gradient_checkpointing True \
    --bf16 True \
    --model_hidden_size 1536 \
    --audio_encoder_hidden_size 1024 \
    --audio_projector_hidden_size 7168 \
    --audio_num_codebook 7 \
    --text_vocab_size 151936 \
    --text_special_tokens 64 \
    --audio_vocab_size 4096 \
    --audio_special_tokens 64 \
    --text_additional "EOT" "PAD_T" "BOT" "ANS_T" "TTS" \
    --audio_additional "EOA" "PAD_A" "BOA" "ANS_A" "ASR" \
    --cache_dir ${CACHE_DIR} \
    --audio_manifest ${AUDIO_MANIFEST} \
    --transcript_file ${TRANS_FILE} \
    --codec_file ${CODEC_FILE} \
    --output_dir ${OUPTUT_DIR} \
    --sample_rate 16000 \
    --audio_feature_rate 50 \
    --dataloader_num_workers 0 \
    --remove_unused_columns False \
