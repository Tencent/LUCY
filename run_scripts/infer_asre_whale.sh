#!/bin/bash
WORK_DIR=$(pwd)

# ME=$(basename "$0")
# ME=${ME%.*}
# TIMESTAMP=$(date '+%m%d%y-%H%M%S')

CACHE_DIR=/mnt/data/hetinggao/models

MANIFEST_DIR=/mnt/data/hetinggao/manifest/SER/stage1

#MODEL_NAME_OR_PATH=/mnt/data/hetinggao/Projects/vita-e2e/outputs/vita_s1v0p1p1_zh
#TEXT_VOCAB_SIZE=151936

MODEL_NAME_OR_PATH=/mnt/data/hetinggao/Projects/vita-e2e/outputs/vita_qwen2-7b-instruct_s1v3_zh
MODEL_NAME_OR_PATH=/mnt/data/hetinggao/Projects/vita-e2e/outputs/vita_qwen2-7b-instruct_s1v6_zh
TEXT_VOCAB_SIZE=152064
CKPT_PATH=$MODEL_NAME_OR_PATH
EXPNAME=$(basename `dirname $MODEL_NAME_OR_PATH`)
CKPTNAME=$(basename $MODEL_NAME_OR_PATH)
SUFFIX=test
# OUTPUT_PATH=$WORK_DIR/generated/$EXPNAME-$CKPTNAME-$SUFFIX
OUTPUT_PATH=$WORK_DIR/generated/$CKPTNAME-best-$SUFFIX
mkdir -p $OUTPUT_PATH
TASKS="ASRE"
SAVE_AUDIO=False
TEXT_ONLY=True

AUDIO_IN="${MANIFEST_DIR}/eval.tsv"
TEXT_IN="${MANIFEST_DIR}/eval.wrdemo"
TEXT_OUT="${MANIFEST_DIR}/eval.wrdemo"
CODEC_OUT="${MANIFEST_DIR}/eval.snac"

export PYTHONPATH=$WORK_DIR
python vita/scripts/infer_asre_whale.py \
    --audio_in ${AUDIO_IN} \
    --text_in ${TEXT_IN} \
    --text_out ${TEXT_OUT} \
    --codec_out ${CODEC_OUT} \
    --audio_feature_rate 50 \
    --sample_rate 16000 \
    --tasks ${TASKS} \
    --model_type "qwen2" \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --audio_encoder /mnt/data/hetinggao/models/audio-encoder-Qwen2-7B-instruct-weight-base-11wh-tunning \
	--audio_projector_type identity \
    --model_hidden_size 1536 \
    --freeze_backbone True \
    --freeze_audio_encoder True \
    --audio_encoder_hidden_size 1024 \
    --audio_projector_hidden_size 7168 \
    --audio_num_codebook 7 \
    --text_vocab_size ${TEXT_VOCAB_SIZE} \
    --text_special_tokens 64 \
    --audio_vocab_size 4096 \
    --audio_special_tokens 64 \
    --cache_dir ${CACHE_DIR} \
    --text_additional "EOT" "PAD_T" "BOT" "ANS_T" "TTS" "TQA" "TQAA" \
    --audio_additional "EOA" "PAD_A" "BOA" "ANS_A" "ASR" "AQA" "AQAA" "M29" "F10" "ER" \
	--max_code_length 100 \
    --max_keep_sample_size $((30*16000)) \
    --ckpt_path ${CKPT_PATH} \
    --output_path ${OUTPUT_PATH} \
	--save_audio ${SAVE_AUDIO} \
	--output_text_only ${TEXT_ONLY} \
	--max_input_length 100000000 \

unused="""
    --shuffle False \
	--emotion ${EMOTION} \
"""
