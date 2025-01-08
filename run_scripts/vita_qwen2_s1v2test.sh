#!/bin/bash
WORK_DIR=$(pwd)

ME=$(basename "$0")
ME=${ME%.*}
TIMESTAMP=$(date '+%m%d%y-%H%M%S')

OUTPUT_DIR=${WORK_DIR}/outputs/${ME}-${TIMESTAMP}
# CACHE_DIR=/data/workspace/.cache
CACHE_DIR=/mnt/data/hetinggao/models
MANIFEST_DIR="/mnt/data/hetinggao/manifest/aishell3/sub1"
# MANIFEST_DIR="/mnt/data/hetinggao/manifest/WenetSpeech"
#MANIFEST_DIR="/mnt/data/hetinggao/manifest/sub1"
# MANIFEST_DIR="/data/workspace/manifest/sub100"

AUDIO_MANIFEST="${MANIFEST_DIR}/train.tsv"
TRANS_FILE="${MANIFEST_DIR}/train.wrd"
CODEC_FILE="${MANIFEST_DIR}/train.snac"
PHONE_FILE="${MANIFEST_DIR}/train.pinyin"
PHONE_DICT="${MANIFEST_DIR}/dict.pinyin.txt"

MODEL_NAME_OR_PATH="Qwen/Qwen2-1.5B"
AUDIO_ENCODER="openai/whisper-medium"
PHONE_TOKENIZER="/mnt/data/hetinggao/manifest/aishell3/pinyin256.json"

export PYTHONPATH=$WORK_DIR
python vita/scripts/train_s1v2.py \
    --model_type "qwen2" \
    --model_name_or_path /mnt/data/hetinggao/models/Qwen2-1.5B \
    --audio_encoder /mnt/data/hetinggao/models/whisper-medium \
    --freeze_audio_encoder True \
    --per_device_train_batch_size 1 \
    --num_train_epochs 5 \
    --save_strategy "steps" \
    --save_steps 10 \
    --save_total_limit 1 \
    --eval_strategy "no" \
    --learning_rate 5e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --logging_steps 2 \
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
    --phone_file ${PHONE_FILE} \
    --phone_dict ${PHONE_DICT} \
    --phone_vocab_size 256 \
    --phone_special_tokens 16 \
    --text_additional "EOT" "PAD_T" "BOT" "ANS_T" "TTS" \
    --audio_additional "EOA" "PAD_A" "BOA" "ANS_A" "ASR" \
    --phone_additional "EOP" "PAD_P" "BOP" "ANS_P" \
    --cache_dir ${CACHE_DIR} \
    --audio_manifest ${AUDIO_MANIFEST} \
    --transcript_file ${TRANS_FILE} \
    --codec_file ${CODEC_FILE} \
    --output_dir ${OUTPUT_DIR} \
    --sample_rate 16000 \
    --audio_feature_rate 50 \
    --dataloader_num_workers 0 \
    --remove_unused_columns False \
    --shuffle False \
	--tune_text_embed True \
	--tie_word_embeddings True \
	--tasks "ASR" \
	--phone_tokenizer ${PHONE_TOKENIZER} \
	
