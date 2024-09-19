#!/bin/bash
WORK_DIR=$(pwd)

ME=$(basename "$0")
ME=${ME%.*}
TIMESTAMP=$(date '+%m%d%y-%H%M%S')

OUTPUT_DIR=${WORK_DIR}/outputs/${ME}-${TIMESTAMP}
CACHE_DIR=/mnt/data/hetinggao/models
MANIFEST_DIR="/mnt/data/hetinggao/manifest/aishell3"
# MANIFEST_DIR="/data/workspace/manifest/sub100"

AUDIO_MANIFEST="${MANIFEST_DIR}/train.tsv"
TRANS_FILE="${MANIFEST_DIR}/train.wrd"
CODEC_FILE="${MANIFEST_DIR}/train.snac"

MODEL_NAME_OR_PATH="Qwen/Qwen2-1.5B"
AUDIO_ENCODER="openai/whisper-medium"

unset CUDA_VISIBLE_DEVICES
export PYTHONPATH=$WORK_DIR
deepspeed --include localhost:4,5,6,7 --master_port 29501 vita/scripts/train_s1.py \
    --deepspeed config/zero2.json\
    --model_type "qwen2" \
    --model_name_or_path /mnt/data/hetinggao/models/Qwen2-1.5B \
    --audio_encoder /mnt/data/hetinggao/models/whisper-medium \
    --freeze_audio_encoder True \
    --per_device_train_batch_size 32 \
    --num_train_epochs 50 \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 10 \
    --eval_strategy "no" \
    --learning_rate 5e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --logging_steps 50 \
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
    --output_dir ${OUTPUT_DIR} \
    --sample_rate 16000 \
    --audio_feature_rate 50 \
    --dataloader_num_workers 5 \
    --remove_unused_columns False \
    --tasks "ASR" \
    --max_keep_sample_size $((30*16000)) \
	--tune_text_embed True \
	--tie_word_embeddings True \
