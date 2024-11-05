import os
import re
import json
import torch
import random
import soundfile as sf
import itertools
import transformers
import numpy as np
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, field
from torch.utils.data import Dataset
from transformers import logging
from copy import deepcopy
from itertools import groupby
from .data_util import load_data
from . import conversation as conversation_lib
from ..constants import IGNORE_INDEX, AUDIO_PH, PAD_TOKEN
from transformers import TrainerCallback

logger = logging.get_logger(__name__)

@dataclass
class DataArguments:
    # Audio Question | Text Question | Codec Answer | Text Answer 
    # audio_in       | text_in       | codec_out    | text_out
    audio_in: Optional[List[str]] = field(default_factory=list)
    text_in: Optional[List[str]] = field(default_factory=list)
    codec_out: Optional[List[str]] = field(default_factory=list)
    text_out: Optional[List[str]] = field(default_factory=list)
    data_jsons: Optional[List[str]] = field(default_factory=list)
    eval_audio_in: Optional[List[str]] = field(default_factory=list)
    eval_text_in: Optional[List[str]] = field(default_factory=list)
    eval_codec_out: Optional[List[str]] = field(default_factory=list)
    eval_text_out: Optional[List[str]] = field(default_factory=list)
    eval_data_jsons: Optional[List[str]] = field(default_factory=list)

    asr_template: Optional[str] = field(default=None)
    data_ratio: Optional[List[float]] = field(default=None)

    max_convs: Optional[int] = field(default=10)
    sample_rate: Optional[int] = field(default=16_000)
    audio_feature_rate: Optional[int] = field(default=50)
    max_keep_sample_size: Optional[int] = field(default=16_000 * 60)
    min_keep_sample_size: Optional[int] = field(default=16_000 * 0)
    num_codebook: Optional[int] = field(default=7)
    text_additional_tokens: Optional[Dict] = field(default=None)
    audio_additional_tokens: Optional[Dict] = field(default=None)
    padded_vocab_size: Optional[int] = field(default=None)
    padded_audio_vocab_size: Optional[int] = field(default=None)
    tasks: Optional[List[str]] = field(default_factory=lambda: ["AQA", "TQA"])
    add_codec_target: Optional[bool] = field(default=True)

def sync_data_args(model_args, data_args):
    setattr(data_args, "num_codebook", model_args.audio_num_codebook)    
    text_additional_tokens = {token: model_args.text_vocab_size + i for i, token in enumerate(model_args.text_additional)}
    audio_additional_tokens = {token: model_args.audio_vocab_size + i for i, token in enumerate(model_args.audio_additional)}
    setattr(data_args, "text_additional_tokens", text_additional_tokens)
    setattr(data_args, "audio_additional_tokens", audio_additional_tokens)
    setattr(data_args, "padded_vocab_size", model_args.text_vocab_size + model_args.text_special_tokens)
    setattr(data_args, "padded_audio_vocab_size", model_args.audio_vocab_size + model_args.audio_special_tokens)

def load_single_turn_data(audio_ins, text_ins, text_outs, codec_outs, max_keep, min_keep, tolerance, max_num_samples=None):
    if codec_outs is None or len(codec_outs) == 0:
        codec_outs = [None] * len(audio_ins)
        assert len(codec_outs) == len(text_ins)
    
    # num_samples_total = 0
    # num_samples_list = []
    # (
    #     audio_paths_cat, lengths_cat, starts_cat, ends_cat, 
    #     textin_cat, textout_offsets_list_cat, codec_offsets_list_cat
    # ) = (
    #     [], [], [], [], [], [], []
    # )
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


        # num_samples_list.append(nsamples)
        # num_samples_total += nsamples
        # audio_paths_cat.append(audio_paths[:nsamples])
        # lengths_cat.append(lengths[:nsamples])
        # starts_cat.append(starts[:nsamples])
        # ends_cat.append(ends[:nsamples])
        # textin_cat.append(textin[:nsamples])
        # textout_offsets_list_cat.append(textout_offsets_list[:nsamples])
        # codec_offsets_list_cat.append(codec_offsets_list[:nsamples] if codec_offsets_list is not None else None)
        d = {
            "audio_paths": audio_paths, "lengths": lengths, 
            "starts": starts, "ends": ends,
            "textin": textin, 
            "text_out": text_out, "textout_offsets_list": textout_offsets_list, 
            "codec_out": codec_out, "codec_offsets_list": codec_offsets_list
        }
        data.append(d)
    return data
    # return (
    #     num_samples_list, num_samples_total, audio_paths_cat, lengths_cat, starts_cat, ends_cat, 
    #     textin_cat, textout_offsets_list_cat, codec_offsets_list_cat
    # )
def load_multi_turn_data(data_jsons):
    data = []
    for data_json in data_jsons:
        if data_json.endswith("json"):
            with open(data_json, "r") as f:
                _d = json.load(f)
            d = []
            for item in _d:
                if "conversations" not in item:
                    item = {"conversations": item}
                d.append(item)
        elif data_json.endswith("jsonl"):
            d = []
            with open(data_json, "r") as f:
                for l in f:
                    item = json.loads(l.strip())
                    if "conversations" not in item:
                        item = {"conversations": item}
                    d.append(item)
        else:
            raise ValueError(f"Can't read {data_json}")
        data.append(d)
    return data

def get_codec(codec_data):
    if type(codec_data) is str:
        with open(codec_data) as f:
            codec = f.readline().strip()
        codec = list(map(int,codec.split()))
    elif type(codec_data) is list:
        codec = codec_data
    else:
        raise ValueError(f"codec_data of type {type(codec_data)} is not implemented: {codec_data}")
    codec = torch.LongTensor(codec)
    return codec

class TA2TADataset(Dataset):
    def __init__(
        self, 
        text_tokenizer: transformers.PreTrainedTokenizer, 
        audio_processor: Any, 
        data_args: DataArguments, 
        split: Optional[str] = "train"):
        super(TA2TADataset, self).__init__()
        self.split = split
        if split == "train":
            self.data_jsons = data_args.data_jsons
            self.audio_in, self.text_in, self.codec_out, self.text_out = (
                data_args.audio_in, data_args.text_in, data_args.codec_out, data_args.text_out, 
            )
        elif split == "eval":
            self.data_jsons = data_args.eval_data_jsons
            self.audio_in, self.text_in, self.codec_out, self.text_out = (
                data_args.eval_audio_in, 
                data_args.eval_text_in, 
                data_args.eval_codec_out, 
                data_args.eval_text_out,
            )
        else:
            raise ValueError(f"{split} is not Implemented")

        self.codec_out = [co if co != "<NONE>" else None for co in self.codec_out]
        self.audio_num_codebook = data_args.num_codebook 

        self.data = []
        self.num_samples_list, self.num_samples = [], 0
        single_turn_data = load_single_turn_data(
            self.audio_in, self.text_in, self.text_out, self.codec_out,
            data_args.max_keep_sample_size, 
            data_args.min_keep_sample_size,
            tolerance=int(0.2 * data_args.sample_rate),
        )
        self.data += single_turn_data
        for d in single_turn_data:
            self.num_samples_list.append(len(d["audio_paths"]))
            self.num_samples += len(d["audio_paths"])

        multi_turn_data = load_multi_turn_data(self.data_jsons)
        self.data += multi_turn_data
        for d in multi_turn_data:
            self.num_samples_list.append(len(d))
            self.num_samples += len(d)

        self.data_ratio = data_args.data_ratio
        self.random_data = not (self.data_ratio is None or len(self.data_ratio) == 0)
        print(self.num_samples_list, self.num_samples, self.data_ratio)
            

        self.tasks = data_args.tasks
        if set(self.tasks).intersection(["ASR", "ASRA"]):
            assert data_args.asr_template is not None
            with open(data_args.asr_template) as f:
                self.asr_template = json.load(f)
                # set start_query, yes_response, end_query, output_response
                for k, v in self.asr_template.items():
                    setattr(self, k, v)
        self.audio_feature_rate = data_args.audio_feature_rate
        self.text_tokenizer = text_tokenizer
        self.audio_processor = audio_processor
        self.data_args = data_args
        self.sample_rate = data_args.sample_rate

        self.text_vocab_size_padded = data_args.padded_vocab_size
        self.audio_vocab_size_padded = data_args.padded_audio_vocab_size

        # "EOT": 151936, "PAD_T": 151937, "BOT": 151938, "ANS_T": 151939, "TTS": 151940
        for tk, idx in data_args.text_additional_tokens.items():
            setattr(self, tk, idx)
        # audio_additional_tokens: { "EOA": 4096, "PAD_A": 4097, "BOA": 4098, "ANS_A": 4099, "ASR": 4100, "AQA": 4101, "AQAA": 4102 } 
        # t + 4160 * i + 152000
        #         | EOA    | PAD_A  | BOA    | ANS_A  | ASR
        # Layer 0 | 156096 | 156097 | 156098 | 156099 | 156100
        # Layer 1 | 160256 | 160257 | 160258 | 160259 | 160260
        # Layer 2 | 164416 | 164417 | 164418 | 164419 | 164420
        # Layer 3 | 168576 | 168577 | 168578 | 168579 | 168580
        # Layer 4 | 172736 | 172737 | 172738 | 172739 | 172740
        # Layer 5 | 176896 | 176897 | 176898 | 176899 | 176900
        # Layer 6 | 181056 | 181057 | 181058 | 181059 | 181060
        # ====================================================
        #         | EOT    | PAD_T  | BOT    | ANS_T  | TTS
        # Layer 7 | 151936 | 151937 | 151938 | 151939 | 151940
        for tk, idx in data_args.audio_additional_tokens.items():
            setattr(self, tk, idx)

        self.AUDIO_PH = self.text_tokenizer.convert_tokens_to_ids(AUDIO_PH)
        self.PAD_TOKEN = self.text_tokenizer.convert_tokens_to_ids(PAD_TOKEN)

        self.tasks_with_audio_input = set(["ASR", "AQA", "AQAA", "ASRA"])
        self.tasks_with_text_input = set(["TTS", "TQA", "TQAA"])
        self.tasks_with_audio_output = set(["AQAA", "TQAA", "TTS", "ASRA", "RQACONVA"])
        self.tasks_with_random_input = set(["RQACONV", "RQACONVA"])
        self.add_codec_target = data_args.add_codec_target
        self.epoch = 0

    def __len__(self):
        return self.num_samples
    
    def extract_source(self, 
        audio_paths, lengths, starts, ends,
        textin, text_out, textout_offsets_list, 
        codec_out, codec_offsets_list,
        task, index
    ):
        audio_path = audio_paths[index]
        audio_length = lengths[index]
        start, end = starts[index], ends[index]
        textin_str = textin[index].replace("\\n", "\n")
        textout_str = self.get_textout(text_out, textout_offsets_list, index).replace("\\n", "\n")
        codec = self.get_codec(codec_out, codec_offsets_list, index)
        
        audio_data = {"wavpath": audio_path, "audio_length": audio_length, "start": start, "end": end}
        if task == "ASR":
            _use_template_random = random.random()
            text_sq, wavpath_sq, _       = random.choice(self.start_query)
            text_or, _,          _       = random.choice(self.output_response)
            text_y,  wavpath_y,  codec_y = random.choice(self.yes_response)
            if _use_template_random > 0.5: # template 1
                convs = [
                    {"role": "user", "wavpath": [wavpath_sq, audio_data]},
                    {"role": "assistant",  "content": text_or + textout_str}
                ]
            else: # template 1
                convs = [
                    {"role": "user", "wavpath": wavpath_sq, "content": text_sq},
                    {"role": "assistant",  "wavpath": wavpath_y, "content": text_y, "codec": codec_y},
                    {"role": "user", "wavpath": audio_data},
                    {"role": "assistant",  "content": text_or + textout_str}
                ]
        else:
            convs = [
                {"role": "user", "content": textin_str, "wavpath": audio_data},
                {"role": "assistant",  "content": textout_str, "codec": codec}
            ]
        source = {"conversations": convs}
        return source

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __getitem__(self, ix) -> Any:
        if self.random_data and self.split == "train":
            random.seed(ix+self.epoch)
            did = random.choices(list(range(len(self.data))), weights=self.data_ratio)[0]
            idx = random.randint(0, self.num_samples_list[did]-1)
        else:
            did, idx = self.get_dataset_idx(ix)

        item = self.get_item(did, idx)
        item["index"] = ix
        return item
        # did, idx = 1, 504749 # contains eos which is same as pad
    def get_item(self, did, idx):
        task = self.tasks[did]
        dataset = self.data[did]
        if "textin" in dataset:
            source = self.extract_source(**dataset, task=task, index=idx)
        else:
            source = deepcopy(dataset[idx])

        try:
            if source["conversations"][0]["role"] not in ["user", "assistant"]:
                source["conversations"] = source["conversations"][1:]
            num_convs = len(source["conversations"])
            num_rounds = num_convs // 2 # retain at least 1 round of conversations
            num_rounds_clipped = random.randint(0, num_rounds-1) if self.split == "train" else 0 # both side inclusive
            # num_rounds_clipped = 1
            # print("num_rounds_clipped", num_convs, num_rounds_clipped, source)

            # import pdb; pdb.set_trace()
            num_convs_remain = min(self.data_args.max_convs, num_convs-num_rounds_clipped*2)
            valid_until = 0
            for i in range(0, num_convs_remain, 2):
                if i == num_convs_remain - 1: # num_convs_remain is odd and is the second last position 
                    break
                if source["conversations"][i]["role"] == "user" and source["conversations"][i+1]["role"] == "assistant":
                    valid_until += 2
            assert valid_until > 0, f"data: {did} {idx} | {num_convs_remain} {source}"
            assert valid_until <= num_convs_remain, f"{valid_until} {num_convs_remain}"

            source["conversations"] = source["conversations"][:valid_until] 
            conversations = source["conversations"]
            # print(conversations)
            data_dict = self.tokenize_conversation(conversations, task)

            # import pdb; pdb.set_trace()

            if task in self.tasks_with_audio_output and "codec" in conversations[-1]:
                codec_data = conversations[-1]["codec"] # last answer
                codec = get_codec(codec_data)
                data_dict["codec"] = codec
            data_dict["task"] = task
            data_dict["did"] = did
            data_dict["idx"] = idx
        except Exception as e:
            print("dataset", did, idx, e)
            print("source", source)
            print("num_rounds_clipped", num_rounds_clipped)
            raise e

        return data_dict
    
    def load_wav(self, wavpath):
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
            assert sr == self.sample_rate, f"Audio sampling rate {sr} != {self.sample_rate}"
            assert audio_length is None or len(wav) == audio_length, \
                f"Audio length {len(wav)} != {audio_length} of {wpath} with start {start} and end {end}"
            assert end is None or (end - start == audio_length), \
                f"Audio length {audio_length} != end {end} - start {start}"
            if i > 0:
                interval = random.uniform(0.35, 0.75)
                si_leng = int(interval * self.sample_rate)
                silence = np.zeros(si_leng)
                wavs.append(silence)
            wavs.append(wav)
        # print("wav leng:", [len(w) for w in wavs])
        wav_cat = np.concatenate(wavs)
        return wav_cat, sr

    def tokenize_conversation(self, conversations, task) -> Dict:
        use_audio_input = task in self.tasks_with_audio_input
        # only use codec as target at the last turn of conversation
        use_codec_output = task in self.tasks_with_audio_output and "codec" in conversations[-1]
        
        conv = conversation_lib.conv_qwen2.copy()
        human, gpt, *addiitonal_roles = conv.roles

        if conversations[0]["role"] != human: # skip the first system prompt
            conversations = conversations[1:]
        if conversations[-1]["role"] != gpt: # skip the first system prompt
            conversations = conversations[:-1]
        assert len(conversations) % 2 == 0, conversations 

        conv.messages, audios, audio_lengths = [], [], []
        for i, sentence in enumerate(conversations):
            
            _use_audio_input_random = random.random() > 0.5
            _use_audio_input = use_audio_input if task not in self.tasks_with_random_input else _use_audio_input_random
            # print(_use_audio_input_random, task, _use_audio_input)

            role = sentence["role"]
            
            assert role in conv.roles[i%2::2], f"{i} {role} not in {conv.roles[i%2::2]}"
            contains_only_wav = "wavpath" in sentence and "content" not in sentence
            contains_only_text = "content" in sentence and "wavpath" not in sentence
            if role == human and not contains_only_text and (contains_only_wav or _use_audio_input):
                wav, sr = self.load_wav(sentence["wavpath"])
                audio_length = len(wav)
                audio = self.audio_processor(wav, sampling_rate=sr, return_tensors="pt").input_features
                audio_length = int(audio_length / self.sample_rate * self.audio_feature_rate) + 1
                message = AUDIO_PH * (audio_length + 2) # 2 for BOA and EOA
                audios.append(audio)
                audio_lengths.append(audio_length)
            elif role == gpt and use_codec_output and i == len(conversations)-1: 
                # add codec only for the last turn of conversation
                # import pdb; pdb.set_trace()
                message = conv.remove_function_call(sentence["content"])
            elif role in conv.roles: 
                # currently only role "user", "assistant", "observation"
                message = sentence["content"]
            else:
                raise ValueError("Edge case encounterd. Debug!")

            conv.append_message(role, message)
        
        conversation, is_target = conv.get_prompt_split_by_target()

        input_ids, labels = [], []
        # print("conversations", conversation)
        for conv, istar in zip(conversation, is_target):
            inp_ids = self.text_tokenizer.encode(conv, return_tensors="pt")[0]
            tar_ids = inp_ids.clone() if istar else inp_ids.new_zeros(inp_ids.shape).fill_(IGNORE_INDEX)
            input_ids.append(inp_ids)
            labels.append(tar_ids)
            # print(istar, f"num placeholder: {(inp_ids==self.AUDIO_PH).sum()}", self.text_tokenizer.decode(inp_ids), )

        input_ids, labels = torch.cat(input_ids), torch.cat(labels)

        data_dict = dict(
            input_ids=input_ids,
            labels=labels,
        )
        if len(audios) > 0:
            assert len(audio_lengths) > 0, audio_lengths
        data_dict["audios"] = audios
        data_dict["audio_lengths"] = audio_lengths
        return data_dict

    def get_dataset_idx(self, idx):
        for dataset_idx, ns in enumerate(self.num_samples_list):
            residual = idx - ns
            if residual < 0:
                break
            idx = residual
        return dataset_idx, idx

    def get_codec(self, codec_out, codec_offsets_list, index):
        if self.codec_out is None or codec_out is None:
            return None
        with open(codec_out, "rb") as f:
            offset_s, offset_e = codec_offsets_list[index]
            f.seek(offset_s)
            codec = f.read(offset_e - offset_s).decode()
        codec_array = list(map(int, codec.split()))
        return codec_array
    
    def get_textout(self, text_out, textout_offsets_list, index):
        with open(text_out, "rb") as f:
            offset_s, offset_e = textout_offsets_list[index]
            f.seek(offset_s)
            tout = f.read(offset_e - offset_s).decode()
        return tout
    
    def extract_input_output(self, batch):
        batched_input_ids, batched_labels, batched_audio_attention_mask = [], [], []
        shifted_PAD_A = torch.LongTensor([self.codec_layer_shift(self.PAD_A, i) for i in range(self.audio_num_codebook)])
        for item in batch:
            input_ids, labels, codec, audio_lengths = (
                item["input_ids"], item["labels"],
                item.get("codec", None), item.get("audio_lengths", [])
            )
            if codec is not None:
                # pad codec
                codec_by_layer = codec.view(-1, self.audio_num_codebook) # T x 7
                codec_by_layer = torch.cat([
                    codec_by_layer, 
                    codec.new_zeros([1,self.audio_num_codebook]).fill_(self.EOA)
                ], dim=0)
                codec_length, _ = codec_by_layer.shape
                codec_length_padded = codec_length + self.audio_num_codebook
                

                # plus 1 for additional EOT token and minus 1 for index of last position
                last_ans_length = len(labels)-1 - torch.where(labels < 0)[0][-1] + 1
                assert last_ans_length > 0, last_ans_length

                num_text_pad = codec_length_padded - last_ans_length # pad last answer to the same length of codec
                assert  num_text_pad > 0, f"{item['did']}, {item['idx']} has num_text_pad {num_text_pad} <= 0 [{codec_length_padded}, {last_ans_length}]"
                last_ans_length = codec_length_padded
                input_ids = torch.cat([
                    input_ids,
                    input_ids.new_zeros(1).fill_(self.EOT), # additional EOT token
                    input_ids.new_zeros(num_text_pad).fill_(self.PAD_T)
                ])
                labels = torch.cat([
                    labels,
                    labels.new_zeros(1).fill_(self.EOT), # additional EOT token
                    labels.new_zeros(num_text_pad).fill_(IGNORE_INDEX)
                ])

                input_codec = codec.new_zeros([len(input_ids), self.audio_num_codebook]).fill_(IGNORE_INDEX)
                input_codec[:,:] = shifted_PAD_A[None,:]
                label_codec = codec.new_zeros([len(input_ids), self.audio_num_codebook]).fill_(IGNORE_INDEX)
                for i in range(self.audio_num_codebook):
                    num_audio_pad_left = i + 1
                    num_audio_pad_right = self.audio_num_codebook - num_audio_pad_left
                    input_codec_i = torch.cat([
                        codec.new_zeros(1).fill_(self.ANS_A),
                        codec.new_zeros(num_audio_pad_left).fill_(self.PAD_A),
                        codec_by_layer[:, i],
                        codec.new_zeros(num_audio_pad_right).fill_(self.PAD_A),
                    ])
                    label_codec_i = torch.cat([
                        codec.new_zeros(num_audio_pad_left).fill_(IGNORE_INDEX),
                        codec_by_layer[:, i],
                        codec.new_zeros(num_audio_pad_right).fill_(IGNORE_INDEX),
                    ])
                    input_codec[-last_ans_length-1:, i] = self.codec_layer_shift(input_codec_i, i)
                    label_codec[-last_ans_length:, i] = label_codec_i
                
            else:
                input_codec = input_ids.new_zeros([len(input_ids), self.audio_num_codebook]).fill_(IGNORE_INDEX)
                input_codec[:,:] = shifted_PAD_A[None,:]
                label_codec = input_ids.new_zeros([len(input_ids), self.audio_num_codebook]).fill_(IGNORE_INDEX)
            
            assert (input_codec > 0).all()

            i_chunk, start, end = 0, 0, 0
            shifted_BOA = torch.LongTensor([self.codec_layer_shift(self.BOA, i) for i in range(self.audio_num_codebook)])
            shifted_EOA = torch.LongTensor([self.codec_layer_shift(self.EOA, i) for i in range(self.audio_num_codebook)])
            audio_attention_mask = input_ids == self.AUDIO_PH
            for is_placeholder, chunk in groupby(audio_attention_mask.clone()):
                chunk_length = len(list(chunk))
                assert chunk_length > 2 # chunk has at least 1 BOA, 1 EOA, and 1 audio token
                end += chunk_length
                if is_placeholder:
                    assert chunk_length == audio_lengths[i_chunk] + 2
                    input_codec[start] = shifted_BOA
                    input_codec[end-1] = shifted_EOA
                    audio_attention_mask[[start,end-1]] = False
                    i_chunk += 1

                start = end
            
            batched_audio_attention_mask.append(audio_attention_mask)
            
            input_ids = torch.cat([input_codec, input_ids.unsqueeze(-1)], dim=-1) # T x 8
            labels = torch.cat([label_codec, labels.unsqueeze(-1)], dim=-1) # T x 8
            batched_input_ids.append(input_ids)
            batched_labels.append(labels)
        return batched_input_ids, batched_labels, batched_audio_attention_mask

    def collate_fn(self, batch):
        """
        audio_lengths = torch.LongTensor(sum([item["audio_lengths"] for item in batch], []))
        if self.split == "train" and audio_lengths.numel() == 0:
            _did = 0
            _idx = random.randint(0, self.num_samples_list[_did]-1)
            _item = self.get_item(_did, _idx)
            _item["index"] = _idx
            batch.append(_item)
        """
        input_ids, labels, audio_attention_mask = self.extract_input_output(batch)
        tasks, indices, dids, idxs = [[item[key] for item in batch] for key in [
            "task", "index", "did", "idx"
        ]]

        audio_lengths = torch.LongTensor(sum([item["audio_lengths"] for item in batch], []))
        # assert audio_lengths.numel() > 0
        
        audios = None
        if audio_lengths.numel() > 0:
            audios = torch.cat(sum([item["audios"] for item in batch], []))
        for iii, ii in enumerate(input_ids):
            if (ii == self.PAD_TOKEN).any():
                print(iii, ii)
                print(dids[iii], idxs[iii])
            assert (ii != self.PAD_TOKEN).all(), f'index {iii} did {dids[iii]} idx {idxs[iii]} input_ids {ii}'
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.PAD_TOKEN
        )
        attention_mask = input_ids[...,-1].ne(self.PAD_TOKEN)
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        audio_attention_mask = torch.nn.utils.rnn.pad_sequence(
            audio_attention_mask, batch_first=True, padding_value=False
        )

        collated = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "audios": audios,
            "audio_lengths": audio_lengths,
            "audio_attention_mask": audio_attention_mask,
            "tasks": tasks,
            "indices": indices, "dids": dids, "idxs": idxs
        }
        return collated
            
    def codec_layer_shift(self, input_id, layer):
        return input_id + self.text_vocab_size_padded + layer * self.audio_vocab_size_padded

class SetEpochCallback(TrainerCallback):
    """
    Trigger re-computing subset for dataset Examples-proportional mixing, see `dataset::ProportionMixingDataset`

    A hack that modifies the train dataset, pointed by Trainer's dataloader
    """

    def on_epoch_begin(self, args, state, control, train_dataloader, **kwargs):
        train_dataloader.dataset.set_epoch(state.epoch)

def make_data_module(text_tokenizer: transformers.PreTrainedTokenizer, audio_processor, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = TA2TADataset(text_tokenizer=text_tokenizer, audio_processor=audio_processor, data_args=data_args)
    data_collator = train_dataset.collate_fn
    eval_dataset = TA2TADataset(text_tokenizer=text_tokenizer, audio_processor=audio_processor, data_args=data_args, split="eval")
    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=data_collator, callbacks=[SetEpochCallback()])
