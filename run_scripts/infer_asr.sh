#!/bin/bash
WORK_DIR=$(pwd)

# ME=$(basename "$0")
# ME=${ME%.*}
# TIMESTAMP=$(date '+%m%d%y-%H%M%S')

CACHE_DIR=/mnt/data/hetinggao/models

# MANIFEST_DIR=/mnt/data/hetinggao/manifest/moss-003-sft-data/sub10
# OUTPUT_PATH=/mnt/data/hetinggao/vita-e2e/generated/zh_tte_s3_72k_eval_3600tk_aqaa
# TASKS="AQAA"

# MANIFEST_DIR=/mnt/data/hetinggao/manifest/moss-003-sft-data/sub10
# OUTPUT_PATH=/mnt/data/hetinggao/vita-e2e/generated/zh_tte_s3_72k_eval_3600tk_tqaa
# TASKS="TQAA"

# MANIFEST_DIR=/mnt/data/hetinggao/manifest/custom_questions
# OUTPUT_PATH=/mnt/data/hetinggao/vita-e2e/generated/zh_tte_s3_72k_eval_3600tk_custom_tqaa
# TASKS="TQAA"

# MANIFEST_DIR=/mnt/data/hetinggao/manifest/WenetSpeech
# OUTPUT_PATH=/mnt/data/hetinggao/vita-e2e/generated/zh_tte_s3_72k_eval_asr/test_net_hyp.txt
# mkdir -p $(dirname $OUTPUT_PATH)
# TASKS="ASR"
# CKPT_PATH=/mnt/data/hetinggao/vita-e2e/outputs/vita_qwen2_s3_zh_parallel_tte-100124-134115/checkpoint-72000

MANIFEST_DIR=/mnt/data/hetinggao/manifest/WenetSpeech/sub10
OUTPUT_PATH=/mnt/data/hetinggao/vita-e2e/generated/zh_tte_s1_tts_40k_eval_asr
mkdir -p $OUTPUT_PATH
TASKS="ASR"
SAVE_AUDIO=True
CKPT_PATH=/mnt/data/hetinggao/vita-e2e/outputs/vita_qwen2_s1_zh_tts-101024-034258/checkpoint-40000

# AUDIO_IN="${MANIFEST_DIR}/train.tsv"
# TEXT_IN="${MANIFEST_DIR}/train.textin"
# TEXT_OUT="${MANIFEST_DIR}/train.textout"
# CODEC_OUT="${MANIFEST_DIR}/train.snac"

AUDIO_IN="${MANIFEST_DIR}/eval.tsv"
TEXT_IN="${MANIFEST_DIR}/eval.wrd"
TEXT_OUT="${MANIFEST_DIR}/eval.wrd"
CODEC_OUT="${MANIFEST_DIR}/eval.snac"

# AUDIO_IN="${MANIFEST_DIR}/test_meeting.tsv"
# TEXT_IN="${MANIFEST_DIR}/test_meeting.wrd"
# TEXT_OUT="${MANIFEST_DIR}/test_meeting.wrd"
# CODEC_OUT="${MANIFEST_DIR}/test_meeting.snac"

# AUDIO_IN="${MANIFEST_DIR}/test_net.tsv"
# TEXT_IN="${MANIFEST_DIR}/test_net.wrd"
# TEXT_OUT="${MANIFEST_DIR}/test_net.wrd"
# CODEC_OUT="${MANIFEST_DIR}/test_net.snac"

MODEL_NAME_OR_PATH="Qwen/Qwen2-1.5B"
AUDIO_ENCODER="openai/whisper-medium"

export PYTHONPATH=$WORK_DIR
python vita/scripts/infer.py \
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
	--max_code_length 100 \
    --max_keep_sample_size $((30*16000)) \
    --ckpt_path ${CKPT_PATH} \
    --output_path ${OUTPUT_PATH} \
	--save_audio ${SAVE_AUDIO} \
