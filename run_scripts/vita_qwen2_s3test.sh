#!/bin/bash
WORK_DIR=$(pwd)

ME=$(basename "$0")
ME=${ME%.*}
TIMESTAMP=$(date '+%m%d%y-%H%M%S')

OUTPUT_DIR=${WORK_DIR}/outputs/${ME}-${TIMESTAMP}

CACHE_DIR=/mnt/data/hetinggao/models
MODEL_NAME_OR_PATH="/mnt/data/hetinggao/vita-e2e/outputs/vita_qwen2_s2_zh_parallel_tte_te_lr5e-5-093024-060627/checkpoint-14000"
AUDIO_ENCODER="openai/whisper-medium"

WENET_DIR="/mnt/data/hetinggao/manifest/WenetSpeech/sub800k"
ALPACA_DIR="/mnt/data/hetinggao/manifest/Alpaca-CH"
MOSS002_DIR="/mnt/data/hetinggao/manifest/moss-002-sft-data/zh_helpfulness"
MOSS003_DIR="/mnt/data/hetinggao/manifest/moss-003-sft-data"
# MANIFEST_DIR="/data/workspace/manifest/sub100"

DATASET_DIRS=(${WENET_DIR} ${ALPACA_DIR} ${ALPACA_DIR} ${MOSS002_DIR} ${MOSS002_DIR} ${MOSS003_DIR} ${MOSS003_DIR})
AUDIO_IN_EXT=(tsv tsv tsv tsv tsv tsv tsv)
TEXT_IN_EXT=(wrd textin textin textin textin textin textin)
TEXT_OUT_EXT=(wrd textout textout textout textout textout textout)
CODEC_OUT_EXT=("<NONE>" "<NONE>" "<NONE>" "<NONE>" "<NONE>" snac snac)
TASKS="ASR AQA TQA AQA TQA AQAA TQAA"

AUDIO_IN=""; TEXT_IN=""; TEXT_OUT=""; CODEC_OUT=""
EVAL_AUDIO_IN=""; EVAL_TEXT_IN=""; EVAL_TEXT_OUT=""; EVAL_CODEC_OUT=""
for i in "${!DATASET_DIRS[@]}"; do
    DDIR=${DATASET_DIRS[$i]}
    AUDIO_IN="$AUDIO_IN $DDIR/train.${AUDIO_IN_EXT[$i]}"
    EVAL_AUDIO_IN="$EVAL_AUDIO_IN $DDIR/eval.${AUDIO_IN_EXT[$i]}"
    TEXT_IN="$TEXT_IN $DDIR/train.${TEXT_IN_EXT[$i]}"
    EVAL_TEXT_IN="$EVAL_TEXT_IN $DDIR/eval.${TEXT_IN_EXT[$i]}"
    TEXT_OUT="$TEXT_OUT $DDIR/train.${TEXT_OUT_EXT[$i]}"
    EVAL_TEXT_OUT="$EVAL_TEXT_OUT $DDIR/eval.${TEXT_OUT_EXT[$i]}"
    if [[ ${CODEC_OUT_EXT[$i]} == "<NONE>" ]]; then 
        CODEC_OUT="$CODEC_OUT <NONE>"
        EVAL_CODEC_OUT="$EVAL_CODEC_OUT <NONE>" 
    else 
        CODEC_OUT="$CODEC_OUT $DDIR/train.${CODEC_OUT_EXT[$i]}"
        EVAL_CODEC_OUT="$EVAL_CODEC_OUT $DDIR/eval.${CODEC_OUT_EXT[$i]}" 
    fi
done
echo "AUDIO_IN": $AUDIO_IN
echo "EVAL_AUDIO_IN": $EVAL_AUDIO_IN
echo "TEXT_IN": $TEXT_IN
echo "EVAL_TEXT_IN": $EVAL_TEXT_IN
echo "TEXT_OUT": $TEXT_OUT
echo "EVAL_TEXT_OUT": $EVAL_TEXT_OUT
echo "CODEC_OUT": $CODEC_OUT
echo "EVAL_CODEC_OUT": $EVAL_CODEC_OUT

export PYTHONPATH=$WORK_DIR
python vita/scripts/train_s3.py \
    --model_type "qwen2" \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --audio_encoder /mnt/data/hetinggao/models/whisper-medium \
    --freeze_backbone False \
    --freeze_audio_encoder_adapter False \
    --freeze_audio_encoder True \
    --freeze_tts_adapter False \
    --freeze_embed_tokens False \
    --per_device_train_batch_size 24 \
    --per_device_eval_batch_size 8 \
    --num_train_epochs 5 \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --eval_strategy "steps" \
    --load_best_model_at_end True \
    --eval_steps 1000 \
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
    --text_additional "EOT" "PAD_T" "BOT" "ANS_T" "TTS" "TQA" "TQAA" \
    --audio_additional "EOA" "PAD_A" "BOA" "ANS_A" "ASR" "AQA" "AQAA" \
    --audio_in ${AUDIO_IN} \
    --text_in ${TEXT_IN} \
    --text_out ${TEXT_OUT} \
    --codec_out ${CODEC_OUT} \
    --eval_audio_in ${EVAL_AUDIO_IN} \
    --eval_text_in ${EVAL_TEXT_IN} \
    --eval_text_out ${EVAL_TEXT_OUT} \
    --eval_codec_out ${EVAL_CODEC_OUT} \
	--tasks ${TASKS} \
    --output_dir ${OUTPUT_DIR} \
    --sample_rate 16000 \
    --audio_feature_rate 50 \
    --dataloader_num_workers 0 \
    --remove_unused_columns False \
    --shuffle False \
	--tune_text_embed True \
	--tie_word_embeddings True \
	--loss_reduction mean \
	
