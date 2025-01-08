#!/bin/bash
WORK_DIR=$(pwd)

ME=$(basename "$0")
ME=${ME%.*}
TIMESTAMP=$(date '+%m%d%y-%H%M%S')

OUTPUT_DIR=${WORK_DIR}/outputs/${ME}-${TIMESTAMP}
CACHE_DIR=/mnt/data/hetinggao/models
MODEL_NAME_OR_PATH=/mnt/data/hetinggao/models/llava-s2-pretrain_video

WENET_DIR="/mnt/data/hetinggao/manifest/WenetSpeech"
ALPACA_DIR="/mnt/data/hetinggao/manifest/Alpaca-CH"
MOSS002_DIR="/mnt/data/hetinggao/manifest/moss-002-sft-data/zh_helpfulness"
MOSS003_DIR="/mnt/data/hetinggao/manifest/moss-003-sft-data"

DATASET_DIRS=(${WENET_DIR})
AUDIO_IN_EXT=(tsv)
TEXT_IN_EXT=(wrd)
TEXT_OUT_EXT=(wrd)
CODEC_OUT_EXT=("snac")
TASKS="ASR"
. $(dirname "$0")/parse_data_dir.sh

#unset CUDA_VISIBLE_DEVICES
export PYTHONPATH=$WORK_DIR
#deepspeed --include localhost:0,1,2,3,4,5,6,7 vita/scripts/train.py \
#    --deepspeed config/zero2.json \
python vita/scripts/train.py \
    --initialize_additional_modules True \
    --model_type "qwen2" \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --audio_encoder /mnt/data/hetinggao/models/whisper-medium \
    --freeze_backbone True \
    --freeze_audio_encoder_adapter False \
    --freeze_audio_encoder True \
    --freeze_tts_adapter False \
    --freeze_embed_tokens False \
	--post_tts_adapter True \
    --per_device_train_batch_size 2 \
    --add_codec_target True \
    --num_train_epochs 50 \
    --save_strategy "steps" \
    --load_best_model_at_end True \
    --save_steps 1000 \
    --save_total_limit 5 \
    --eval_strategy "steps" \
    --learning_rate 5e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --logging_steps 50 \
    --lr_scheduler_type "cosine" \
    --gradient_checkpointing True \
    --bf16 True \
    --post_tts_adapter_num_layers 7 \
    --model_hidden_size 3584 \
    --audio_encoder_hidden_size 1024 \
    --audio_projector_hidden_size 7168 \
    --audio_num_codebook 7 \
    --text_vocab_size 152064 \
    --text_special_tokens 64 \
    --audio_vocab_size 4096 \
    --audio_special_tokens 64 \
    --text_additional "EOT" "PAD_T" "BOT" "ANS_T" "TTS" "TQA" "TQAA" \
    --audio_additional "EOA" "PAD_A" "BOA" "ANS_A" "ASR" "AQA" "AQAA" \
    --cache_dir ${CACHE_DIR} \
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
    --dataloader_num_workers 5 \
    --remove_unused_columns False \
    --max_keep_sample_size $((30*16000)) \
	--tune_text_embed True \
	--tie_word_embeddings True \
    --loss_reduction mean \
