#!/bin/bash
WORK_DIR=$(pwd)

# ME=$(basename "$0")
# ME=${ME%.*}
# TIMESTAMP=$(date '+%m%d%y-%H%M%S')

CACHE_DIR=/mnt/data/hetinggao/models

SAVE_AUDIO=True
TEXT_ONLY=False

# MODEL_NAME_OR_PATH=/mnt/data/hetinggao/vita-e2e/outputs/vita_qwen2_s4_zh-110224-020715/checkpoint-34000
MODEL_NAME_OR_PATH=/mnt/data/hetinggao/vita-e2e/outputs/vita_qwen2_s3v2p2_zh/checkpoint-34500
MODEL_NAME_OR_PATH=/mnt/data/hetinggao/vita-e2e/outputs/vita_qwen2_s3v2p2p2_zh/checkpoint-32500
MODEL_NAME_OR_PATH=/mnt/data/hetinggao/vita-e2e/backups/vita_qwen2_s3v2p2p2_zh/checkpoint-32500
MODEL_NAME_OR_PATH=/mnt/data/hetinggao/vita-e2e/outputs/vita_qwen2_s3v2p2p2_zh/checkpoint-136000
MODEL_NAME_OR_PATH=/mnt/data/hetinggao/vita-e2e/outputs/vita_qwen2_s3v2p2p2_zh/checkpoint-51500
MODEL_NAME_OR_PATH=/mnt/data/hetinggao/vita-e2e/outputs/vita_qwen2_s3v2p2p2_zh
MODEL_NAME_OR_PATH=/mnt/data/hetinggao/Projects/vita-e2e/outputs/vita_qwen2_s3v2p5_zh/checkpoint-26500
# MODEL_NAME_OR_PATH=/mnt/data/hetinggao/vita-e2e/outputs/vita_qwen2_s4_zh_v4/checkpoint-43400
AUDIO_ENCODER="/mnt/data/hetinggao/models/whisper-medium"

EXPNAME=$(basename `dirname $MODEL_NAME_OR_PATH`)
CKPTNAME=$(basename $MODEL_NAME_OR_PATH)
SUFFIX=test
OUTPUT_PATH=$WORK_DIR/generated/$EXPNAME-$CKPTNAME-$SUFFIX
# OUTPUT_PATH=$WORK_DIR/generated/$CKPTNAME-best-$SUFFIX

mkdir -p $OUTPUT_PATH

export PYTHONPATH=$WORK_DIR
python vita/scripts/demo.py \
    --audio_feature_rate 50 \
    --sample_rate 16000 \
    --model_type "qwen2" \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --audio_encoder $AUDIO_ENCODER \
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
	--max_code_length 1000 \
    --max_keep_sample_size $((25*16000)) \
    --output_path ${OUTPUT_PATH} \
	--save_audio ${SAVE_AUDIO} \
	--output_text_only ${TEXT_ONLY} \

unused="""
    --audio_in ${AUDIO_IN} \
    --text_in ${TEXT_IN} \
    --text_out ${TEXT_OUT} \
    --codec_out ${CODEC_OUT} \
	
"""
