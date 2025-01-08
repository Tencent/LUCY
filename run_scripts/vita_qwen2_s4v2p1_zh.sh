#!/bin/bash
WORK_DIR=$(pwd)

ME=$(basename "$0")
ME=${ME%.*}
TIMESTAMP=$(date '+%m%d%y-%H%M%S')

OUTPUT_DIR=${WORK_DIR}/outputs/${ME}

CACHE_DIR=/mnt/data/hetinggao/models
MODEL_NAME_OR_PATH="/mnt/data/hetinggao/Projects/vita-e2e/outputs/vita_qwen2_s3v4p2p1p1_zh"
AUDIO_ENCODER="/mnt/data/hetinggao/models/whisper-medium"

WENET_DIR="/mnt/data/hetinggao/manifest/WenetSpeech"
ALPACA_DIR="/mnt/data/hetinggao/manifest/Alpaca-CH"
MOSS002_DIR="/mnt/data/hetinggao/manifest/moss-002-sft-data/zh_helpfulness"
MOSS003_DIR="/mnt/data/hetinggao/manifest/moss-003-sft-data"

UNION60W_DIR="/mnt/data/hetinggao/manifest/union180w_kcg60w"
COMMON_DIR="/mnt/data/hetinggao/manifest/vita2tts_v3/jsons/common"
FC_DIR="/mnt/data/hetinggao/manifest/vita2tts_v3/jsons/fc"
FCNEG_DIR="/mnt/data/hetinggao/manifest/vita2tts_v3/jsons/fcneg"
NATURAL_DIR="/mnt/data/hetinggao/manifest/vita2tts_v3/jsons/natural"
EMO1_DIR="/mnt/data/hetinggao/manifest/SER/stage3/1"
EMO2_DIR="/mnt/data/hetinggao/manifest/SER/stage3/2"
NEGA_DIR="/mnt/data/hetinggao/manifest/vita_negative"

DATASET_DIRS=(${MOSS003_DIR} ${MOSS003_DIR})
AUDIO_IN_EXT=(tsv tsv)
TEXT_IN_EXT=(textin textin)
TEXT_OUT_EXT=(textout textout)
CODEC_OUT_EXT=(snac snac)
DATA_JSONS="$NATURAL_DIR/train.json $NATURAL_DIR/train.json $FC_DIR/train.json $FC_DIR/train.json $COMMON_DIR/train.json $COMMON_DIR/train.json $UNION60W_DIR/train.json $EMO1_DIR/train.json $EMO1_DIR/train.json $EMO2_DIR/train.json $EMO2_DIR/train.json"
EVAL_DATA_JSONS="$NATURAL_DIR/eval.json $NATURAL_DIR/eval.json $FC_DIR/eval.json $FC_DIR/eval.json $COMMON_DIR/eval.json $COMMON_DIR/eval.json $UNION60W_DIR/eval.json $EMO1_DIR/eval.json $EMO1_DIR/eval.json $EMO2_DIR/eval.json $EMO2_DIR/eval.json"
TASKS="RQACONVA_NTRL RQACONV_NTRL RQACONVA RQACONV RQACONVA RQACONV RQACONV RQACONV_EMO RQACONVA_EMO RQACONV_EMO RQACONVA_EMO"
NEGATIVE_TSVS="$NEGA_DIR/all.tsv"
NEGATIVE_RATIO=0.25
#DATA_JSONS="$FC_DIR/train.json"
#EVAL_DATA_JSONS="$FC_DIR/eval.json"
#TASKS="RQACONVA"

. $(dirname "$0")/parse_data_dir.sh

unset CUDA_VISIBLE_DEVICES
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=$WORK_DIR
# deepspeed --include localhost:0 vita/scripts/train_v5p1.py \
deepspeed --include localhost:0,1,2,3,4,5,6,7 vita/scripts/train_v5p1.py \
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
	--max_input_length 2000 \
    --data_jsons $DATA_JSONS \
	--negative_tsvs $NEGATIVE_TSVS \
	--negative_ratio $NEGATIVE_RATIO \
    --eval_data_jsons $EVAL_DATA_JSONS \
    --text_additional "EOT" "PAD_T" "BOT" "ANS_T" "TTS" "TQA" "TQAA" \
    --audio_additional "EOA" "PAD_A" "BOA" "ANS_A" "ASR" "AQA" "AQAA" "M29" "F10" "ER" \
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
    --audio_in ${AUDIO_IN} \
    --text_in ${TEXT_IN} \
    --text_out ${TEXT_OUT} \
    --codec_out ${CODEC_OUT} \
    --eval_audio_in ${EVAL_AUDIO_IN} \
    --eval_text_in ${EVAL_TEXT_IN} \
    --eval_text_out ${EVAL_TEXT_OUT} \
    --eval_codec_out ${EVAL_CODEC_OUT} \
"""
