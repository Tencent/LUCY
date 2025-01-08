#!/bin/bash
WORK_DIR=$(pwd)

# ME=$(basename "$0")
# ME=${ME%.*}
# TIMESTAMP=$(date '+%m%d%y-%H%M%S')

CACHE_DIR=/mnt/data/hetinggao/models


MANIFEST_DIR=/mnt/data/hetinggao/manifest/moss-003-sft-data/sub10
# OUTPUT_PATH=/mnt/data/hetinggao/vita-e2e/generated/zh_s4_tconva_v2_34k_eval
TASKS="TCONVA"
#SAVE_AUDIO=False
#TEXT_ONLY=True
SAVE_AUDIO=True
TEXT_ONLY=False
# CKPT_PATH=/mnt/data/hetinggao/vita-e2e/outputs/vita_qwen2_s4_zh_ngpu-110224-090936/checkpoint-41000
# CKPT_PATH=/mnt/data/hetinggao/vita-e2e/outputs/vita_qwen2_s4_zh-110224-020715/checkpoint-34000
CKPT_PATH=/mnt/data/hetinggao/vita-e2e/backups/vita_qwen2_s3v2p2p2_zh/checkpoint-32500
CV2TTS_DIR="/mnt/data/hetinggao/manifest/cosyvoice2tts/jsons/train_eval_v2"
# DATA_JSONS=/mnt/data/hetinggao/manifest/cosyvoice2tts/jsons/train_eval/eval.json
EXPNAME=$(basename `dirname $CKPT_PATH`)
CKPTNAME=$(basename $CKPT_PATH)
SUFFIX=eval
DATA_JSONS=$CV2TTS_DIR/eval.json
OUTPUT_PATH=$WORK_DIR/generated/$EXPNAME-$CKPTNAME-$SUFFIX
mkdir -p $OUTPUT_PATH

AUDIO_IN="${MANIFEST_DIR}/eval.tsv"
TEXT_IN="${MANIFEST_DIR}/eval.textin"
TEXT_OUT="${MANIFEST_DIR}/eval.textout"
CODEC_OUT="<NONE>"

MODEL_NAME_OR_PATH="Qwen/Qwen2-1.5B"
AUDIO_ENCODER="openai/whisper-medium"

export PYTHONPATH=$WORK_DIR
python vita/scripts/infer_v2.py \
    --audio_feature_rate 50 \
    --sample_rate 16000 \
    --tasks ${TASKS} \
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
    --max_keep_sample_size $((25*16000)) \
    --ckpt_path ${CKPT_PATH} \
    --output_path ${OUTPUT_PATH} \
	--save_audio ${SAVE_AUDIO} \
	--output_text_only ${TEXT_ONLY} \
	--data_jsons ${DATA_JSONS}

unused="""
    --audio_in ${AUDIO_IN} \
    --text_in ${TEXT_IN} \
    --text_out ${TEXT_OUT} \
    --codec_out ${CODEC_OUT} \
	
"""
echo output to $OUTPUT_PATH
