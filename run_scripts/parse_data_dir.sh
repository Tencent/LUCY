#!/bin/bash


# function parse_data_dir() {
    # export AUDIO_IN=""; export TEXT_IN=""; export TEXT_OUT=""; export CODEC_OUT=""
    # export EVAL_AUDIO_IN=""; export EVAL_TEXT_IN=""; export EVAL_TEXT_OUT=""; export EVAL_CODEC_OUT=""
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
    # echo "AUDIO_IN": $AUDIO_IN
    # echo "EVAL_AUDIO_IN": $EVAL_AUDIO_IN
    # echo "TEXT_IN": $TEXT_IN
    # echo "EVAL_TEXT_IN": $EVAL_TEXT_IN
    # echo "TEXT_OUT": $TEXT_OUT
    # echo "EVAL_TEXT_OUT": $EVAL_TEXT_OUT
    # echo "CODEC_OUT": $CODEC_OUT
    # echo "EVAL_CODEC_OUT": $EVAL_CODEC_OUT
# }