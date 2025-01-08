#!/bin/bash
WORK_DIR=$(pwd)

# ME=$(basename "$0")
# ME=${ME%.*}
# TIMESTAMP=$(date '+%m%d%y-%H%M%S')

CACHE_DIR=/mnt/data/hetinggao/models

# MANIFEST_DIR=/mnt/data/hetinggao/manifest/WenetSpeech
MANIFEST_DIR=/mnt/data/hetinggao/manifest/SER/metadata/sub10

MODEL_NAME_OR_PATH=/mnt/data/hetinggao/Projects/vita-e2e/outputs/vita_qwen2_s1v3_zh/checkpoint-55000
CKPT_PATH=$MODEL_NAME_OR_PATH
EXPNAME=$(basename `dirname $MODEL_NAME_OR_PATH`)
CKPTNAME=$(basename $MODEL_NAME_OR_PATH)
SUFFIX=test
OUTPUT_PATH=$WORK_DIR/generated/$EXPNAME-$CKPTNAME-$SUFFIX
# OUTPUT_PATH=$OUTPUT_DIR/hyp.txt
mkdir -p $OUTPUT_PATH
TASKS="ASRE"
SAVE_AUDIO=False
TEXT_ONLY=True
# MODEL_NAME_OR_PATH=/mnt/data/hetinggao/models/Qwen2-1.5B-Instruct

AUDIO_IN="${MANIFEST_DIR}/eval.tsv"
TEXT_IN="${MANIFEST_DIR}/eval.wrd"
TEXT_OUT="${MANIFEST_DIR}/eval.wrd"
CODEC_OUT="${MANIFEST_DIR}/eval.snac"
EMOTION="${MANIFEST_DIR}/eval.emotion"

# AUDIO_IN="${MANIFEST_DIR}/test_meeting.tsv"
# TEXT_IN="${MANIFEST_DIR}/test_meeting.wrd"
# TEXT_OUT="${MANIFEST_DIR}/test_meeting.wrd"
# CODEC_OUT="${MANIFEST_DIR}/test_meeting.snac"

# AUDIO_IN="${MANIFEST_DIR}/test_net.tsv"
# TEXT_IN="${MANIFEST_DIR}/test_net.wrd"
# TEXT_OUT="${MANIFEST_DIR}/test_net.wrd"
# CODEC_OUT="${MANIFEST_DIR}/test_net.snac"

export PYTHONPATH=$WORK_DIR
python vita/scripts/infer_asre.py \
    --audio_in ${AUDIO_IN} \
    --text_in ${TEXT_IN} \
    --text_out ${TEXT_OUT} \
    --codec_out ${CODEC_OUT} \
	--emotion ${EMOTION} \
    --audio_feature_rate 50 \
    --sample_rate 16000 \
    --tasks ${TASKS} \
    --model_type "qwen2" \
    --model_name_or_path $MODEL_NAME_OR_PATH \
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
	--max_code_length 100 \
    --max_keep_sample_size $((30*16000)) \
    --ckpt_path ${CKPT_PATH} \
    --output_path ${OUTPUT_PATH} \
	--save_audio ${SAVE_AUDIO} \
	--output_text_only ${TEXT_ONLY} \

unused="""
    --shuffle False \
"""
