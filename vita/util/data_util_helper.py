import os
import json
import torch
import random
import soundfile as sf
import numpy as np

def get_audio_encoder_type(audio_encoder):
    if "whisper" in audio_encoder.lower():
        audio_encoder_type = "whisper" 
    elif "audio-encoder-qwen2-7b-instruct" in audio_encoder.lower():
        audio_encoder_type = "whale"
    else:
        raise ValueError(f"Unknown encoder type {model_args.audio_encoder}")
    return audio_encoder_type

def sync_data_args(model_args, data_args):
    setattr(data_args, "num_codebook", model_args.audio_num_codebook)    
    text_additional_tokens = {token: model_args.text_vocab_size + i for i, token in enumerate(model_args.text_additional)}
    audio_additional_tokens = {token: model_args.audio_vocab_size + i for i, token in enumerate(model_args.audio_additional)}
    setattr(data_args, "text_additional_tokens", text_additional_tokens)
    setattr(data_args, "audio_additional_tokens", audio_additional_tokens)
    setattr(data_args, "padded_vocab_size", model_args.text_vocab_size + model_args.text_special_tokens)
    setattr(data_args, "padded_audio_vocab_size", model_args.audio_vocab_size + model_args.audio_special_tokens)
    audio_encoder_type = get_audio_encoder_type(model_args.audio_encoder)
    setattr(data_args, "audio_encoder_type", audio_encoder_type)

# Modified from https://github.com/facebookresearch/fairseq/blob/main/fairseq/data/audio/hubert_dataset.py
def load_audio(manifest_path, transcript_path, max_keep, min_keep, tolerance=0):
    n_long, n_short = 0, 0
    names, inds, sizes, audio_frames, trans, starts, ends = [], [], [], [], [], [], []
    n_mismatch = 0
    with open(manifest_path) as f, open(transcript_path) as ftrans:
        root = f.readline().strip()
        for ind, (line, line_trans) in enumerate(zip(f, ftrans)):
            items = line.strip().split("\t")
            assert len(items) == 2 or len(items) == 4, line
            start, end = 0, None
            if len(items) == 4:
                name, frames, start, end = items
                start, end, frames = int(start), int(end), int(frames)
                if end > frames:
                    assert end - frames < tolerance, f"length difference of {name} {end} - {frames} = {end-frames} > tolerance of {tolerance} frames"
                    logger.info(f"set audio end from {end} to {frames} for {name} with difference of {end-frames}")
                    n_mismatch += 1
                end = min(end, frames)
                sz = end - start
            elif len(items) == 2:
                name, frames = items
                frames = int(frames)
                sz = int(frames)
            else:
                raise ValueError(f"Case of {len(items)} items are not implemented")
            if min_keep is not None and sz < min_keep:
                n_short += 1
            elif max_keep is not None and sz > max_keep:
                n_long += 1
            else:
                names.append(os.path.join(root, items[0]))
                inds.append(ind)
                sizes.append(sz)
                audio_frames.append(frames)
                trans.append(line_trans.strip())
                starts.append(start)
                ends.append(end)
    tot = ind + 1
    logger.warning(
        (
            f"max_keep={max_keep}, min_keep={min_keep}, "
            f"loaded {len(names)}, skipped {n_short} short and {n_long} long, "
            f"longest-loaded={max(sizes)}, shortest-loaded={min(sizes)}, "
            f"find {n_mismatch} mismatch utterence in {len(inds)} / {tot} samples"
        )
    )
    return names, inds, tot, sizes, audio_frames, trans, starts, ends

def load_single_turn_data(audio_ins, text_ins, text_outs, codec_outs, max_keep, min_keep, tolerance, max_num_samples=None):
    if codec_outs is None or len(codec_outs) == 0:
        codec_outs = [None] * len(audio_ins)
        assert len(codec_outs) == len(text_ins)
    data = []
    for audio_in, text_in, text_out, codec_out in zip(audio_ins, text_ins, text_outs, codec_outs):
        num_samples, audio_paths, lengths, starts, ends, textin, codec_offsets_list, textout_offsets_list = load_data(
            audio_in, text_in, text_out, codec_out, max_keep, min_keep, tolerance
        )
        nsamples = min(max_num_samples, num_samples) if max_num_samples is not None else num_samples
        audio_paths = audio_paths[:nsamples]
        lengths = lengths[:nsamples]
        starts = starts[:nsamples]
        ends = ends[:nsamples]
        textin = textin[:nsamples]
        textout_offsets_list = textout_offsets_list[:nsamples]
        codec_offsets_list = codec_offsets_list[:nsamples] if codec_offsets_list is not None else None

        d = {
            "audio_paths": audio_paths, "lengths": lengths, 
            "starts": starts, "ends": ends,
            "textin": textin, 
            "text_out": text_out, "textout_offsets_list": textout_offsets_list, 
            "codec_out": codec_out, "codec_offsets_list": codec_offsets_list
        }
        data.append(d)
    return data

def longest_conv_with_codec(conversation):
    for i, sentence in enumerate(reversed(conversation)):
        if sentence["role"] == "assistant" and "codec" in sentence:
            conv = conversation[:len(conversation)-i]
            # import pdb; pdb.set_trace()
            return conv
    return None

def remove_conv_without_codec(data):
    nd = []
    for item in data:
        conv = longest_conv_with_codec(item["conversations"])
        if conv is not None:
            nd.append({"conversations": conv})
    return nd

def remove_conv_without_audio(data):
    nd = []
    for item in data:
        all_has_audio = True
        conv = item["conversations"]
        for source in conv:
            if source["role"] == "user" and "wavpath" not in source:
                all_has_audio = False
                break
        if all_has_audio:
            nd.append({"conversations": conv})
    return nd

def load_json(data_json):
    with open(data_json, "r") as f:
        _d = json.load(f)
    d = []
    for item in _d:
        if "conversations" not in item:
            item = {"conversations": item}
        d.append(item)
    return d

def load_jsonl(data_json):
    d = []
    with open(data_json, "r") as f:
        for l in f:
            item = json.loads(l.strip())
            if "conversations" not in item:
                item = {"conversations": item}
            d.append(item)
    return d

def load_multi_turn_data(data_jsons, data_codecs, has_audio_outputs, has_audio_inputs):
    if data_codecs is None or len(data_codecs) == 0:
        data_codecs = [None] * len(data_jsons)
    else:
        data_codecs = [dc if dc != "<NONE>" else None  for dc in data_codecs]

    data = []
    for data_json, data_codec, has_audio_output, has_audio_input in zip(
        data_jsons, data_codecs, has_audio_outputs, has_audio_inputs
    ):
        print(f"loading {data_json} and {data_codec}...")
        if data_json.endswith("json"):
            d = load_json(data_json)
        elif data_json.endswith("jsonl"):
            d = load_jsonl(data_json)
        else:
            raise ValueError(f"Can't read {data_json}")

        if has_audio_output:
            ori_size = len(d)
            d = remove_conv_without_codec(d)
            new_size = len(d)
            print(f"removing conv without codec output {data_json} {new_size} samples remains out of {ori_size}")

        if has_audio_input:
            ori_size = len(d)
            d = remove_conv_without_audio(d)
            new_size = len(d)
            print(f"removing conv without audio input {data_json} {new_size} samples remains out of {ori_size}")

        codec_dict = None
        if data_codec is not None:
            with open(data_codec) as f:
                codec_dict = json.load(f)
        data.append([d, codec_dict])
    return data

def load_negative_data(data_tsvs):
    negative_data = []
    for data_tsv in data_tsvs:
        with open(data_tsv) as f:
            root_dir = f.readline().strip()
            data = []
            for l in f:
                rel_path, length = l.strip().split("\t")
                data.append(os.path.join(root_dir, rel_path))
        negative_data.extend(data)
    return negative_data

def get_codec(codec_data, codec_dict=None):
    if type(codec_data) is str:
        if codec_dict is None:
            with open(codec_data) as f:
                codec = f.readline().strip()
        else:
            codec = codec_dict[codec_data]
        codec = list(map(int, codec.split()))
    elif type(codec_data) is list:
        codec = codec_data
    else:
        raise ValueError(f"codec_data of type {type(codec_data)} is not implemented: {codec_data}")
    codec = torch.LongTensor(codec)
    return codec



def load_wav(wavpath, sample_rate=16_000):
    wavpaths = [wavpath] if type(wavpath) is not list else wavpath
    
    wavs = []
    for i, wdata in enumerate(wavpaths):
        if type(wdata) is dict:
            wpath, start, end, audio_length = \
                wdata["wavpath"], wdata["start"], wdata["end"], wdata["audio_length"]
        else:
            wpath, start, end, audio_length = wdata, 0, None, None
        wav, sr = sf.read(wpath, start=start, stop=end)
        if wav.ndim == 2:
            wav = wav.mean(-1)
        assert sr == sample_rate, f"Audio sampling rate {sr} != {sample_rate}"
        assert audio_length is None or len(wav) == audio_length, \
            f"Audio length {len(wav)} != {audio_length} of {wpath} with start {start} and end {end}"
        assert end is None or (end - start == audio_length), \
            f"Audio length {audio_length} != end {end} - start {start}"
        if i > 0:
            interval = random.uniform(0.35, 0.75)
            si_leng = int(interval * sample_rate)
            silence = np.zeros(si_leng)
            wavs.append(silence)
        wavs.append(wav)
    wav_cat = np.concatenate(wavs)
    return wav_cat, sr
