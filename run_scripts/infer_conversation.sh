#!/bin/bash
WORK_DIR=$(pwd)

CACHE_DIR=/mnt/data/hetinggao/models

MANIFEST_DIR=/mnt/data/hetinggao/manifest/custom_questions
OUTPUT_PATH=/mnt/data/hetinggao/vita-e2e/generated/zh_tte_s3_72k_eval_conv/hyp.txt
mkdir -p $(dirname $OUTPUT_PATH)
TASKS="TQAA"

CKPT_PATH=/mnt/data/hetinggao/vita-e2e/outputs/vita_qwen2_s3_zh_parallel_tte-100124-134115/checkpoint-72000

AUDIO_IN="${MANIFEST_DIR}/conversation.tsv"
TEXT_IN="${MANIFEST_DIR}/conversation.textin"
TEXT_OUT="${MANIFEST_DIR}/conversation.textout"
CODEC_OUT="${MANIFEST_DIR}/conversation.snac"

MODEL_NAME_OR_PATH="Qwen/Qwen2-1.5B"
AUDIO_ENCODER="openai/whisper-medium"

export PYTHONPATH=$WORK_DIR
python vita/scripts/infer_conv.py \
    --audio_in ${AUDIO_IN} \
    --text_in ${TEXT_IN} \
    --text_out ${TEXT_OUT} \
    --codec_out ${CODEC_OUT} \
    --audio_feature_rate 50 \
    --sample_rate 16000 \
    --tasks ${TASKS} \
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
    --text_additional "EOT" "PAD_T" "BOT" "ANS_T" "TTS" "TQA" "TQAA" \
    --audio_additional "EOA" "PAD_A" "BOA" "ANS_A" "ASR" "AQA" "AQAA" \
    --max_code_length 360 \
    --max_keep_sample_size $((30*16000)) \
    --ckpt_path ${CKPT_PATH} \
    --output_path ${OUTPUT_PATH} \
