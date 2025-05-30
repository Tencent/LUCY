#!/bin/bash
WORK_DIR=$(pwd)

# ME=$(basename "$0")
# ME=${ME%.*}
# TIMESTAMP=$(date '+%m%d%y-%H%M%S')

CACHE_DIR=/mnt/data/hetinggao/models

SAVE_AUDIO=True
TEXT_ONLY=False

# MODEL_NAME_OR_PATH=/mnt/data/hetinggao/vita-e2e/outputs/vita_qwen2_s4_zh_v4/checkpoint-43400
# MODEL_NAME_OR_PATH=/mnt/data/hetinggao/Projects/vita-e2e/outputs/vita_qwen2_s3v4p2_zh
# MODEL_NAME_OR_PATH=/mnt/data/hetinggao/Projects/vita-e2e/outputs/vita_qwen2_s3v4p2p1_zh/checkpoint-70000
# MODEL_NAME_OR_PATH=/mnt/data/hetinggao/Projects/vita-e2e/outputs/vita_qwen2_s3v4p2p1p1_zh
# MODEL_NAME_OR_PATH=/mnt/data/hetinggao/Projects/vita-e2e/outputs/vita_qwen2-7b-instruct_s4_zh/checkpoint-12000
# MODEL_NAME_OR_PATH=/mnt/data/hetinggao/Projects/vita-e2e/outputs/vita_qwen2-7b-instruct_s4p1_zh/checkpoint-12000
# MODEL_NAME_OR_PATH=/mnt/data/hetinggao/Projects/vita-e2e/outputs/vita_qwen2-7b-instruct_s4v2_zh
# MODEL_NAME_OR_PATH=/mnt/data/hetinggao/Projects/vita-e2e/outputs/vita_qwen2-7b-instruct_s4v2p1_zh
MODEL_NAME_OR_PATH=/mnt/data/hetinggao/Projects/vita-e2e/outputs/vita_qwen2-7b-instruct_s4v2p2_zh
AUDIO_ENCODER=/mnt/data/hetinggao/models/audio-encoder-Qwen2-7B-instruct-weight-base-11wh-tunning

EXPNAME=$(basename `dirname $MODEL_NAME_OR_PATH`)
CKPTNAME=$(basename $MODEL_NAME_OR_PATH)
SUFFIX=testneg
# OUTPUT_PATH=$WORK_DIR/generated/$EXPNAME-$CKPTNAME-$SUFFIX
OUTPUT_PATH=$WORK_DIR/generated/$CKPTNAME-best-$SUFFIX

mkdir -p $OUTPUT_PATH

export PYTHONPATH=$WORK_DIR
python vita/scripts/demo_emofcneg_whale.py \
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
    --text_vocab_size 152064 \
    --text_special_tokens 64 \
    --audio_vocab_size 4096 \
    --audio_special_tokens 64 \
    --cache_dir ${CACHE_DIR} \
    --text_additional "EOT" "PAD_T" "BOT" "ANS_T" "TTS" "TQA" "TQAA" \
    --audio_additional "EOA" "PAD_A" "BOA" "ANS_A" "ASR" "AQA" "AQAA" "F10" "M29" "ER" \
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
