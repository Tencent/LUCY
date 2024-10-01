#!/bin/bash
WORK_DIR=$(pwd)

ME=$(basename "$0")
ME=${ME%.*}
TIMESTAMP=$(date '+%m%d%y-%H%M%S')

OUTPUT_DIR=${WORK_DIR}/outputs/${ME}-${TIMESTAMP}
OUTPUT_DIR=/mnt/data/hetinggao/vita-e2e/outputs/vita_qwen2_s2_zh_parallel_tte_te_lr5e-5-093024-060627

CACHE_DIR=/mnt/data/hetinggao/models
ALPACA_DIR="/mnt/data/hetinggao/manifest/Alpaca-CH"
WENET_DIR="/mnt/data/hetinggao/manifest/WenetSpeech/sub800k"
MOSS_HELPFUL_DIR="/mnt/data/hetinggao/manifest/moss-002-sft-data/zh_helpfulness"
# MANIFEST_DIR="/data/workspace/manifest/sub100"

AUDIO_IN="${ALPACA_DIR}/train.tsv ${ALPACA_DIR}/train.tsv ${MOSS_HELPFUL_DIR}/train.tsv ${MOSS_HELPFUL_DIR}/train.tsv ${WENET_DIR}/train.tsv"
TEXT_IN="${ALPACA_DIR}/train.textin ${ALPACA_DIR}/train.textin ${MOSS_HELPFUL_DIR}/train.textin ${MOSS_HELPFUL_DIR}/train.textin ${WENET_DIR}/train.wrd"
TEXT_OUT="${ALPACA_DIR}/train.textout ${ALPACA_DIR}/train.textout ${MOSS_HELPFUL_DIR}/train.textout ${MOSS_HELPFUL_DIR}/train.textout ${WENET_DIR}/train.wrd"

EVAL_AUDIO_IN="${ALPACA_DIR}/eval.tsv ${ALPACA_DIR}/eval.tsv ${MOSS_HELPFUL_DIR}/eval.tsv ${MOSS_HELPFUL_DIR}/eval.tsv ${WENET_DIR}/eval.tsv"
EVAL_TEXT_IN="${ALPACA_DIR}/eval.textin ${ALPACA_DIR}/eval.textin ${MOSS_HELPFUL_DIR}/eval.textin ${MOSS_HELPFUL_DIR}/eval.textin ${WENET_DIR}/eval.wrd"
EVAL_TEXT_OUT="${ALPACA_DIR}/eval.textout ${ALPACA_DIR}/eval.textout ${MOSS_HELPFUL_DIR}/eval.textout ${MOSS_HELPFUL_DIR}/eval.textout ${WENET_DIR}/eval.wrd"

TASKS="AQA TQA AQA TQA ASR"

MODEL_NAME_OR_PATH="/mnt/data/hetinggao/vita-e2e/backups/vita_qwen2_s1_zh_parallel_tte-091724-063343/checkpoint-50000"
AUDIO_ENCODER="openai/whisper-medium"

unset CUDA_VISIBLE_DEVICES
export PYTHONPATH=$WORK_DIR
deepspeed --include localhost:0,1,2,3,4,5,6,7 vita/scripts/train_s2.py \
    --deepspeed config/zero2.json\
    --model_type "qwen2" \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --audio_encoder /mnt/data/hetinggao/models/whisper-medium \
    --freeze_backbone False \
    --freeze_tts_adapter False \
    --freeze_audio_encoder True \
    --freeze_audio_encoder_adapter True \
    --freeze_embed_tokens False \
    --per_device_train_batch_size 32 \
    --add_codec_target False \
    --num_train_epochs 50 \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 5 \
    --eval_strategy "steps" \
    --load_best_model_at_end True \
    --eval_steps 1000 \
    --learning_rate 5e-5 \
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
    --text_additional "EOT" "PAD_T" "BOT" "ANS_T" "TTS" "TQA" \
    --audio_additional "EOA" "PAD_A" "BOA" "ANS_A" "ASR" "AQA" \
    --cache_dir ${CACHE_DIR} \
    --audio_in ${AUDIO_IN} \
    --text_in ${TEXT_IN} \
    --text_out ${TEXT_OUT} \
    --eval_audio_in ${EVAL_AUDIO_IN} \
    --eval_text_in ${EVAL_TEXT_IN} \
    --eval_text_out ${EVAL_TEXT_OUT} \
    --tasks ${TASKS} \
    --output_dir ${OUTPUT_DIR} \
    --sample_rate 16000 \
    --audio_feature_rate 50 \
    --dataloader_num_workers 5 \
    --remove_unused_columns False \
    --max_keep_sample_size $((30*16000)) \
	--tune_text_embed True \
	--tie_word_embeddings True \
	--loss_reduction mean \
    
