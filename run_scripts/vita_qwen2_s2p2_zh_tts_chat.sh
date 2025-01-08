#!/bin/bash
WORK_DIR=$(pwd)

ME=$(basename "$0")
ME=${ME%.*}
TIMESTAMP=$(date '+%m%d%y-%H%M%S')

OUTPUT_DIR=${WORK_DIR}/outputs/${ME}
CACHE_DIR=/mnt/data/hetinggao/models
MODEL_NAME_OR_PATH=/mnt/data/hetinggao/vita-e2e/outputs/vita_qwen2_s2_zh_tts_chat
AUDIO_ENCODER=/mnt/data/hetinggao/models/whisper-medium

WENET_DIR="/mnt/data/hetinggao/manifest/WenetSpeech/sub800k"
ALPACA_DIR="/mnt/data/hetinggao/manifest/Alpaca-CH"
MOSS002_DIR="/mnt/data/hetinggao/manifest/moss-002-sft-data/zh_helpfulness"
MOSS003_DIR="/mnt/data/hetinggao/manifest/moss-003-sft-data"
CV2TTS_DIR="/mnt/data/hetinggao/manifest/cosyvoice2tts/jsons/train_eval_v2"
FUNCALL_DIR="/mnt/data/hetinggao/manifest/cosyvoice2tts/jsons/funcall_train_eval_v2"
UNION180W_DIR="/mnt/data/hetinggao/manifest/union180w_kcg60w"

DATASET_DIRS=($WENET_DIR)
AUDIO_IN_EXT=(tsv)
TEXT_IN_EXT=(wrd)
TEXT_OUT_EXT=(wrd)
CODEC_OUT_EXT=("<NONE>")
DATA_JSONS="$CV2TTS_DIR/train.json $UNION180W_DIR/train.json"
DATA_CODECS="<NONE> <NONE>"
EVAL_DATA_JSONS="$CV2TTS_DIR/eval.json $UNION180W_DIR/eval.json"
EVAL_DATA_CODECS="<NONE> <NONE>"
TASKS="ASR RQACONV RQACONV"
# DATA_RATIO="0.01 0.02 0.02 0.45 0.3 0.3"
. $(dirname "$0")/parse_data_dir.sh

unset CUDA_VISIBLE_DEVICES
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=$WORK_DIR
deepspeed --include localhost:0,1,2,3,4,5,6,7 vita/scripts/train_v2.py \
    --deepspeed config/zero2.json\
    --model_type "qwen2" \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --audio_encoder $AUDIO_ENCODER \
    --freeze_backbone False \
    --freeze_audio_encoder_adapter True \
    --freeze_audio_encoder True \
    --freeze_tts_adapter False \
    --freeze_embed_tokens False \
    --per_device_train_batch_size 12 \
    --per_device_eval_batch_size 12 \
    --add_codec_target True \
    --num_train_epochs 2 \
    --load_best_model_at_end True \
    --save_steps 500 \
    --save_total_limit 3 \
    --eval_strategy "steps" \
    --eval_steps 500 \
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
    --text_vocab_size 151936 \
    --text_special_tokens 64 \
    --audio_vocab_size 4096 \
    --audio_special_tokens 64 \
    --audio_in ${AUDIO_IN} \
    --text_in ${TEXT_IN} \
    --text_out ${TEXT_OUT} \
    --codec_out ${CODEC_OUT} \
    --eval_audio_in ${EVAL_AUDIO_IN} \
    --eval_text_in ${EVAL_TEXT_IN} \
    --eval_text_out ${EVAL_TEXT_OUT} \
    --eval_codec_out ${EVAL_CODEC_OUT} \
    --data_jsons $DATA_JSONS \
    --data_codecs $DATA_CODECS \
    --eval_data_jsons $EVAL_DATA_JSONS \
    --eval_data_codecs $EVAL_DATA_CODECS \
    --text_additional "EOT" "PAD_T" "BOT" "ANS_T" "TTS" "TQA" "TQAA" \
    --audio_additional "EOA" "PAD_A" "BOA" "ANS_A" "ASR" "AQA" "AQAA" \
    --asr_template /mnt/data/hetinggao/manifest/asr_prompts/asr_template.json \
    --tasks ${TASKS} \
    --output_dir ${OUTPUT_DIR} \
    --sample_rate 16000 \
    --audio_feature_rate 50 \
    --dataloader_num_workers 2 \
    --remove_unused_columns False \
    --max_keep_sample_size $((25*16000)) \
    --tune_text_embed True \
    --tie_word_embeddings True \
    --loss_reduction mean \

unused="""
    --data_ratio $DATA_RATIO \
	    
"""
