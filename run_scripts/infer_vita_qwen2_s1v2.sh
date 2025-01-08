#!/bin/bash
WORK_DIR=$(pwd)

# ME=$(basename "$0")
# ME=${ME%.*}
# TIMESTAMP=$(date '+%m%d%y-%H%M%S')

CACHE_DIR=/mnt/data/hetinggao/models
MANIFEST_DIR="/mnt/data/hetinggao/manifest/aishell3/sub10"
# CKPT_PATH=/mnt/data/hetinggao/vita-e2e/outputs/vita_qwen2_s1v2_zh_parallel_tte_aishell3-091924-161352/checkpoint-70000
# CKPT_PATH=outputs/vita_qwen2_s1v2_zh_parallel_tte_aishell3-092024-163952/checkpoint-24000
CKPT_PATH=/mnt/data/hetinggao/vita-e2e/outputs/vita_qwen2_s1v2_zh_parallel_tte_aishell3-092024-163952/checkpoint-15000
OUTPUT_PATH=/mnt/data/hetinggao/vita-e2e/generated/zh_aishell3_tte_15k_s1v2
mkdir -p $OUTPUT_PATH

AUDIO_MANIFEST="${MANIFEST_DIR}/train.tsv"
TRANS_FILE="${MANIFEST_DIR}/train.wrd"
CODEC_FILE="${MANIFEST_DIR}/train.snac"
PHONE_FILE="${MANIFEST_DIR}/train.pinyin"
PHONE_DICT="${MANIFEST_DIR}/dict.pinyin.txt"

MODEL_NAME_OR_PATH="Qwen/Qwen2-1.5B"
AUDIO_ENCODER="openai/whisper-medium"

PHONE_TOKENIZER="/mnt/data/hetinggao/manifest/aishell3/pinyin256.json"

export PYTHONPATH=$WORK_DIR
python vita/scripts/infer_s1v2.py \
    --audio_manifest ${AUDIO_MANIFEST} \
    --transcript_file ${TRANS_FILE} \
    --codec_file ${CODEC_FILE} \
    --audio_feature_rate 50 \
    --sample_rate 16000 \
    --tasks "ASR" \
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
    --phone_vocab_size 256 \
    --phone_special_tokens 16 \
    --phone_file ${PHONE_FILE} \
    --phone_dict ${PHONE_DICT} \
    --cache_dir ${CACHE_DIR} \
    --text_additional "EOT" "PAD_T" "BOT" "ANS_T" "TTS" \
    --audio_additional "EOA" "PAD_A" "BOA" "ANS_A" "ASR" \
    --phone_additional "EOP" "PAD_P" "BOP" "ANS_P" \
	--max_code_length 360 \
    --ckpt_path ${CKPT_PATH} \
    --output_path ${OUTPUT_PATH} \
	--phone_tokenizer ${PHONE_TOKENIZER} \
    
    

    
    
    # --per_device_train_batch_size 32 \
    # --num_train_epochs 50 \
    # --save_strategy "steps" \
    # --save_steps 1000 \
    # --save_total_limit 10 \
    # --eval_strategy "no" \
    # --learning_rate 5e-4 \
    # --weight_decay 0. \
    # --warmup_ratio 0.03 \
    # --logging_steps 50 \
    # --lr_scheduler_type "cosine" \
    # --gradient_checkpointing True \
    # --bf16 True \

    
    
    
    # --output_dir ${OUPTUT_DIR} \
    
    # --dataloader_num_workers 4 \
    # --remove_unused_columns False \
    
