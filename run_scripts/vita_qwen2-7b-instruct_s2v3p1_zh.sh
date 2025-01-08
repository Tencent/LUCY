#!/bin/bash
WORK_DIR=$(pwd)

ME=$(basename "$0")
ME=${ME%.*}
TIMESTAMP=$(date '+%m%d%y-%H%M%S')

OUTPUT_DIR=${WORK_DIR}/outputs/${ME}

CACHE_DIR=/mnt/data/hetinggao/models
MODEL_NAME_OR_PATH="/mnt/data/hetinggao/Projects/vita-e2e/outputs/vita_qwen2-7b-instruct_s1v3_zh"
# MODEL_NAME_OR_PATH="/mnt/data/hetinggao/vita-e2e/outputs/vita_qwen2_s2v2_zh/previous/.checkpoint-1000"
AUDIO_ENCODER="/mnt/data/hetinggao/models/audio-encoder-Qwen2-7B-instruct-weight-base-11wh-tunning"

WENET_DIR="/mnt/data/hetinggao/manifest/WenetSpeech"
ALPACA_DIR="/mnt/data/hetinggao/manifest/Alpaca-CH"
MOSS002_DIR="/mnt/data/hetinggao/manifest/moss-002-sft-data/zh_helpfulness"
MOSS003_DIR="/mnt/data/hetinggao/manifest/moss-003-sft-data"

DATASET_DIRS=(${WENET_DIR})
AUDIO_IN_EXT=(tsv)
TEXT_IN_EXT=(wrd)
TEXT_OUT_EXT=(wrd)
CODEC_OUT_EXT=("<NONE>")
DATA_JSONS="/mnt/data/hetinggao/manifest/union180w_kcg60w/train.json /mnt/data/hetinggao/manifest/SER/stage2/train1.json /mnt/data/hetinggao/manifest/SER/stage2/train2.json"
EVAL_DATA_JSONS="/mnt/data/hetinggao/manifest/union180w_kcg60w/eval.json /mnt/data/hetinggao/manifest/SER/stage2/eval1.json /mnt/data/hetinggao/manifest/SER/stage2/eval2.json"
TASKS="RQACONV RQACONV_EMO RQACONV_EMO"
# TASKS="ASR"vita_qwen2_s3v4p2p1_zh

. $(dirname "$0")/parse_data_dir.sh

unset CUDA_VISIBLE_DEVICES
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=$WORK_DIR
# deepspeed --include localhost:0 vita/scripts/train_v4p2.py \
deepspeed --include localhost:0,1,2,3,4,5,6,7 vita/scripts/train_v4p2.py \
	--deepspeed config/zero2.json \
    --model_type "qwen2" \
	--initialize_additional_modules False \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --audio_encoder $AUDIO_ENCODER \
	--audio_projector_type "identity" \
    --freeze_backbone True \
    --freeze_audio_encoder_adapter True \
    --freeze_audio_encoder True \
    --freeze_tts_adapter False \
    --freeze_embed_tokens False \
    --per_device_train_batch_size 12 \
    --per_device_eval_batch_size 12 \
    --add_codec_target True \
	--num_train_epochs 5 \
    --load_best_model_at_end True \
    --save_steps 400 \
    --save_total_limit 3 \
    --eval_strategy "steps" \
    --eval_steps 200 \
    --learning_rate 2e-5 \
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
    --text_vocab_size 152064 \
    --text_special_tokens 64 \
    --audio_vocab_size 4096 \
    --audio_special_tokens 64 \
    --text_additional "EOT" "PAD_T" "BOT" "ANS_T" "TTS" "TQA" "TQAA" \
    --audio_additional "EOA" "PAD_A" "BOA" "ANS_A" "ASR" "AQA" "AQAA" "M10" "F29" "ER" \
    --asr_template /mnt/data/hetinggao/manifest/asr_prompts/asr_template.json \
    --tasks ${TASKS} \
    --data_jsons $DATA_JSONS \
    --eval_data_jsons $EVAL_DATA_JSONS \
    --output_dir ${OUTPUT_DIR} \
    --sample_rate 16000 \
    --audio_feature_rate 50 \
    --dataloader_num_workers 2 \
    --remove_unused_columns False \
    --max_keep_sample_size $((25*16000)) \
	--tune_text_embed True \
	--tie_word_embeddings True \
	--loss_reduction mean \
	--max_input_length 1500 \
    
unused="""
	--data_ratio $DATA_RATIO \
    --audio_in ${AUDIO_IN} \
    --text_in ${TEXT_IN} \
    --text_out ${TEXT_OUT} \
    --codec_out ${CODEC_OUT} \
    --eval_audio_in ${EVAL_AUDIO_IN} \
    --eval_text_in ${EVAL_TEXT_IN} \
    --eval_text_out ${EVAL_TEXT_OUT} \
    --eval_codec_out ${EVAL_CODEC_OUT} \
"""
