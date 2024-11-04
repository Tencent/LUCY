import os
import torch
import random
import soundfile as sf
import itertools
import transformers
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, field
from torch.utils.data import Dataset
from transformers import logging

logger = logging.get_logger(__name__)

@dataclass
class DataArguments:
    # Audio Question | Text Question | Codec Answer | Text Answer 
    # audio_in       | text_in       | codec_out    | text_out
    audio_in: Optional[List[str]] = field(default_factory=list)
    text_in: Optional[List[str]] = field(default_factory=list)
    codec_out: Optional[List[str]] = field(default=None)
    text_out: Optional[List[str]] = field(default_factory=list)
    eval_audio_in: Optional[List[str]] = field(default_factory=list)
    eval_text_in: Optional[List[str]] = field(default_factory=list)
    eval_codec_out: Optional[List[str]] = field(default=None)
    eval_text_out: Optional[List[str]] = field(default_factory=list)
    shuffle: Optional[bool] = field(default=True)
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

# https://github.com/facebookresearch/fairseq/blob/main/fairseq/data/audio/hubert_dataset.py
def load_label_offset(label_path, inds, tot):
    with open(label_path, "rb") as f:
        code_lengths = [len(line) for line in f]

        assert (
            len(code_lengths) == tot
        ), f"number of labels does not match ({len(code_lengths)} != {tot})"
        offsets = list(itertools.accumulate([0] + code_lengths))
        offsets = [(offsets[i], offsets[i + 1]) for i in inds]
    return offsets


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

def load_data(audio_in, text_in, text_out, codec_out, max_keep, min_keep, tolerance):
    audio_paths, ixs, tot, lengths, audio_frames, textin, starts, ends = load_audio(
        audio_in, text_in, max_keep, min_keep, tolerance
    )
    num_samples = len(ixs)
    assert num_samples == len(audio_paths)
    assert num_samples == len(lengths)
    assert num_samples == len(audio_frames)
    assert num_samples == len(textin)
    assert num_samples == len(starts)
    assert num_samples == len(ends)

    codec_offsets_list = None
    if codec_out is not None:
        codec_offsets_list = load_label_offset(codec_out, ixs, tot)
        assert num_samples == len(codec_offsets_list)
    textout_offsets_list = load_label_offset(text_out, ixs, tot)
    return num_samples, audio_paths, lengths, starts, ends, textin, codec_offsets_list, textout_offsets_list

def load_concatenated_data(audio_ins, text_ins, text_outs, codec_outs, max_keep, min_keep, tolerance):
    has_codec_out = True
    if codec_outs is None or len(codec_outs) == 0:
        has_codec_out = False
        codec_outs = [None] * len(audio_ins)
        assert len(codec_outs) == len(text_ins)
    
    num_samples_total = 0
    num_samples_list = []
    (
        audio_paths_cat, lengths_cat, starts_cat, ends_cat, 
        textin_cat, textout_offsets_list_cat, codec_offsets_list_cat
    ) = (
        [], [], [], [], [], [], []
    )
    for audio_in, text_in, text_out, codec_out in zip(audio_ins, text_ins, text_outs, codec_outs):
        num_samples, audio_paths, lengths, starts, ends, textin, codec_offsets_list, textout_offsets_list = load_data(
            audio_in, text_in, text_out, codec_out, max_keep, min_keep, tolerance
        )
        num_samples_list.append(num_samples)
        num_samples_total += num_samples
        audio_paths_cat += audio_paths
        lengths_cat += lengths
        starts_cat += starts
        ends_cat += ends
        textin_cat += textin
        textout_offsets_list_cat.append(textout_offsets_list)
        codec_offsets_list_cat.append(codec_offsets_list)
    return (
        num_samples_list, num_samples_total, audio_paths_cat, lengths_cat, starts_cat, ends_cat, 
        textin_cat, textout_offsets_list_cat, codec_offsets_list_cat
    )


class TA2TADataset(Dataset):
    
    def __init__(
        self, 
        text_tokenizer: transformers.PreTrainedTokenizer, 
        audio_processor: Any, 
        data_args: DataArguments, 
        split: Optional[str] = "train"):
        super(TA2TADataset, self).__init__()
        if split == "train":
            self.audio_in, self.text_in, self.codec_out, self.text_out = (
                data_args.audio_in, data_args.text_in, data_args.codec_out, data_args.text_out, 
            )
        elif split == "eval":
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
        (
            self.num_samples_list, self.num_samples, 
            self.audio_paths, self.lengths, self.starts, self.ends,
            self.textin, self.textout_offsets_list, self.codec_offsets_list
        ) = load_concatenated_data(
            self.audio_in, self.text_in, self.text_out, self.codec_out,
            data_args.max_keep_sample_size, 
            data_args.min_keep_sample_size,
            tolerance=int(0.2 * data_args.sample_rate)
        )

        self.ix2ix = list(range(self.num_samples))
        if data_args.shuffle:
            random.shuffle(self.ix2ix)
        
        self.audio_feature_rate = data_args.audio_feature_rate
        self.text_tokenizer = text_tokenizer
        self.audio_processor = audio_processor
        self.data_args = data_args
        self.tasks = data_args.tasks
        self.sample_rate = data_args.sample_rate

        self.padded_vocab_size = data_args.padded_vocab_size
        self.padded_audio_vocab_size = data_args.padded_audio_vocab_size

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
        
        self.tasks_with_audio_input = set(["ASR", "AQA", "AQAA"])
        self.tasks_with_text_input = set(["TTS", "TQA", "TQAA"])
        self.add_codec_target = data_args.add_codec_target

    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, index) -> Any:
        ix = self.ix2ix[index] # random shuffle if any
        did, idx = self.get_dataset_idx(ix)
        task = self.tasks[did]

        textin_str = self.textin[ix].replace("\\n", "\n")
        textout_str = self.get_textout(did, idx).replace("\\n", "\n")
        codec = self.get_codec(did, idx)

        audio_path = self.audio_paths[ix]
        audio_length = self.lengths[ix]
        start, end = self.starts[ix], self.ends[ix]
        wav, sr = sf.read(audio_path, start=start, stop=end)
        if wav.ndim == 2:
            wav = wav.mean(-1)
        assert sr == self.sample_rate, f"Audio sampling rate {sr} != {self.sample_rate}"
        assert len(wav) == audio_length, f"Audio length {len(wav)} != {audio_length} of {audio_path} with start {start} and end {end}"
        assert end is None or (end - start == audio_length), f"Audio length {audio_length} != end {end} - start {start}" 
        audio = self.audio_processor(wav, sampling_rate=sr, return_tensors="pt").input_features 
        audio_length = int(audio_length / self.sample_rate * self.audio_feature_rate) + 1

        textin = self.text_tokenizer([textin_str], return_tensors="pt").input_ids.squeeze(0)
        assert len(textin) > 0, f"{textin} {task} {textin_str}"
        textout = self.text_tokenizer([textout_str], return_tensors="pt").input_ids.squeeze(0)
        data_dict = {
            "audio": audio,
            "audio_length": audio_length,
            "textin": textin,
            "textout": textout,
            "codec": codec,
            "task": task,
            "index": ix, "did": did, "idx": idx # For debugging purpose
        }
        return data_dict

    def get_dataset_idx(self, idx):
        for dataset_idx, ns in enumerate(self.num_samples_list):
            residual = idx - ns
            if residual < 0:
                break
            idx = residual
        return dataset_idx, idx

    def get_codec(self, did, idx):
        if self.codec_out is None or self.codec_out[did] is None:
            return None
        with open(self.codec_out[did], "rb") as f:
            offset_s, offset_e = self.codec_offsets_list[did][idx]
            f.seek(offset_s)
            codec = f.read(offset_e - offset_s).decode()
        codec_array = list(map(int, codec.split()))
        return codec_array
    
    def get_textout(self, did, idx):
        with open(self.text_out[did], "rb") as f:
            offset_s, offset_e = self.textout_offsets_list[did][idx]
            f.seek(offset_s)
            textout = f.read(offset_e - offset_s).decode()
        return textout
    
    def extract_output(self, batch):
        outputs, output_lengths = [], []
        for item in batch:
            codec = item["codec"]
            text = item["textout"]
            text_length = len(text)
            if codec is not None:
                codec = torch.LongTensor(codec)
                codec_length_all = len(codec)
                codec_length = int(codec_length_all / self.audio_num_codebook)
                assert codec_length * self.audio_num_codebook == codec_length_all, \
                    f"length of codec {codec_length_all} is not a multiple of {self.audio_num_codebook}"
                # e.g., [0,1,2,3,...,20] -> [[0,7,14], [1,8,15], ...] (T => 7 x T/7)
                codec_by_layer = codec.view(-1, self.audio_num_codebook).T
                # 1st layer 1 PAD_A, ..., 7th 7 <PAD_A>, plus 1 <EOA> token
                codec_length_padded = codec_length + self.audio_num_codebook + 1

                codec_by_layer_padded = torch.zeros([
                    self.audio_num_codebook, # 7 layers
                    codec_length_padded
                ], dtype=torch.long).fill_(self.PAD_A)

                output_lengs = []
                for i in range(self.audio_num_codebook):
                    num_audio_pad_left = i + 1
                    num_audio_pad_right = self.audio_num_codebook - num_audio_pad_left
                    codec_by_layer_padded[i, num_audio_pad_left:-(num_audio_pad_right+1)] = codec_by_layer[i]
                    codec_by_layer_padded[i, -num_audio_pad_right-1] = self.EOA
                    codec_by_layer_padded[i] = self.codec_layer_shift(codec_by_layer_padded[i], i)
                    # sanity check the length of valid codecs
                    assert codec_length_padded - num_audio_pad_right == codec_length + num_audio_pad_left + 1
                    output_lengs.append(codec_length_padded - num_audio_pad_right)

                assert text_length < codec_length_padded, \
                    f"Number of audio codes {codec_length_padded}" \
                    f"is not enough to cover texts of length {text_length}"
                num_text_pad = codec_length_padded - text_length - 1 # 1 for EOS token
                
                text_padded = torch.cat([
                    text, torch.LongTensor([self.EOT]), 
                    torch.zeros(num_text_pad, dtype=torch.long).fill_(self.PAD_T)]).unsqueeze(0) # 1 x T/7
                output_lengs.append(text_length + 1)
            else:
                output_lengs = [0] * self.audio_num_codebook
                codec_by_layer_padded = torch.zeros(
                    self.audio_num_codebook, text_length+1, dtype=torch.long
                ) # 1 for EOT
                for i in range(self.audio_num_codebook):
                    codec_by_layer_padded[i] = self.codec_layer_shift(self.PAD_A, i)
                    
                text_padded = torch.cat([
                    text, torch.LongTensor([self.EOT]), 
                ]).unsqueeze(0) # 1 x T/7
                output_lengs.append(text_length + 1)
            
            output = torch.cat([codec_by_layer_padded, text_padded], dim=0) # 8 x T/7
            outputs.append(output) # [8 x T / 7]

            
            output_lengths.append(output_lengs) # B x 8
            assert output.shape[-1] == max(output_lengs)

        return outputs, output_lengths


    def extract_input(self, batch):
        inputs, input_lengths = [], []
        for item in batch:
            text, audio_length, task = item["textin"], item["audio_length"], item["task"]
            
            if task in self.tasks_with_audio_input: # collate audio only
                codec_by_layer_padded = torch.zeros(
                    self.audio_num_codebook, audio_length+3, dtype=torch.long
                ) # 1 for BOA, 1 for EOA and 1 for ASR/AQA
                for i in range(self.audio_num_codebook):
                    codec_by_layer_padded[i] = torch.LongTensor(
                        [
                            self.codec_layer_shift(self.BOA, i)
                        ] + [
                            self.codec_layer_shift(self.PAD_A, i)
                        ] * audio_length + [
                            self.codec_layer_shift(self.EOA, i),
                            self.codec_layer_shift(getattr(self, task), i),
                        ]
                    )
                text_padded = torch.cat([
                    torch.LongTensor([self.PAD_T] * (audio_length+2)), # 1 for BOA and 1 for EOA
                    torch.LongTensor([self.ANS_T])
                ]).unsqueeze(0) # 8 x (audio_length+3)
                input_length = audio_length + 3 # 1 for BOA, 1 for EOA and 1 for ASR
            elif task in self.tasks_with_text_input: # collate text only
                codec_by_layer_padded = torch.zeros(
                    self.audio_num_codebook, len(text)+3, dtype=torch.long
                ) # 1 for BOT and 1 for EOT and 1 for ANS
                for i in range(self.audio_num_codebook):
                    codec_by_layer_padded[i] = torch.LongTensor(
                        [
                            self.codec_layer_shift(self.PAD_A, i)
                        ] * (len(text)+2) + [ # 1 for BOT and 1 for EOT
                            self.codec_layer_shift(self.ANS_A, i)
                        ]
                    )
                text_padded = torch.cat([
                    torch.LongTensor([self.BOT]), 
                    text, 
                    torch.LongTensor([self.EOT, getattr(self, task)])]
                ).unsqueeze(0) # 1 x (T+3)
                input_length = len(text) + 3
            else:
                raise ValueError(f"Task {task} is not implemented")
            input = torch.cat([codec_by_layer_padded, text_padded], dim=0) # 8 x input_length

            inputs.append(input)
            input_lengths.append(input_length)
        return inputs, input_lengths

    def collate_fn(self, batch):
        inputs, input_lengths = self.extract_input(batch)
        outputs, output_lengths = self.extract_output(batch)
        tasks = [item["task"] for item in batch]
        codecs = [item["codec"] for item in batch]
        indices = [item["index"] for item in batch]
        dids = [item["did"] for item in batch]
        idxs = [item["idx"] for item in batch]
        audio_lengths = torch.LongTensor([
            item["audio_length"] for item in batch if item["task"] in self.tasks_with_audio_input
        ])
        
        audios, use_audio_indices = None, None
        if len(audio_lengths) > 0:
            audios = torch.cat([item["audio"] for item in batch if item["task"] in self.tasks_with_audio_input]) # B' x 80 x 3000
        else:
            audios = batch[0]["audio"].new(0,*batch[0]["audio"].shape[1:])
        use_audio_indices = torch.LongTensor([
            i for i, task in enumerate(tasks) if task in self.tasks_with_audio_input
        ])

        input_ids, lengs = [], []
        
        for inp, inp_leng, outp, outp_lengs in zip(inputs, input_lengths, outputs, output_lengths):
            input_ids.append(torch.cat([inp, outp], dim=-1).T) # B x T x L
            lengs.append(inp_leng + max(outp_lengs)) # B x L
        input_ids, attention_mask = self.collate_sequence(
            input_ids, max(lengs), pad=0
        )

        output_labels_attention_mask = torch.zeros(input_ids.shape, dtype=bool) # B x T x L
        output_logits_attention_mask = torch.zeros(input_ids.shape, dtype=bool)
        for bi, (inp_leng, outp_lengs, codec) in enumerate(zip(input_lengths, output_lengths, codecs)):
            has_codec_target = codec is not None
            # only set True for the last text layer
            for li, outp_leng in enumerate(outp_lengs):
                layer_type = "audio" if li < self.audio_num_codebook else "text"
                layer_shift = (li + 1) % (self.audio_num_codebook + 1) # pad li+1 for audio layer li=0,1,...,6 and pad 0 for text layer li=7
                if (self.add_codec_target and has_codec_target and layer_type == "audio") or layer_type == "text":
                    output_labels_attention_mask[bi, inp_leng+layer_shift:inp_leng+outp_leng, li] = True
                    output_logits_attention_mask[bi, inp_leng+layer_shift-1:inp_leng+outp_leng-1, li] = True

        collated = {
            "input_ids": input_ids, 
            "attention_mask": attention_mask,
            "audios": audios,
            "audio_lengths": audio_lengths,
            "use_audio_indices": use_audio_indices,
            "output_labels_attention_mask": output_labels_attention_mask, # B x (T_in+T_out) x L
            "output_logits_attention_mask": output_logits_attention_mask, # B x (T_in+T_out) x L
            "tasks": tasks,
            "indices": indices, "dids": dids, "idxs": idxs
        }
        return collated
            
    def collate_sequence(self, inputs, input_length, pad=0.0):
        input_shape = list(inputs[0].shape[1:])
        collated_inputs = inputs[0].new_zeros([len(inputs), input_length]+input_shape)
        attention_mask = (
            torch.BoolTensor(len(inputs), input_length).fill_(True)
        )
        for i, inp in enumerate(inputs):
            diff = len(inp) - input_length
            if diff == 0:
                collated_inputs[i] = inp
            elif diff < 0:
                collated_inputs[i] = torch.cat([inp, inp.new_full([-diff,]+input_shape, pad)])
                attention_mask[i, diff:] = False
            else:
                raise ValueError(f"This should never happen: input length {len(inp)} > longest input length {input_length}")

        return collated_inputs, attention_mask

    def codec_layer_shift(self, input_id, layer):
        return input_id + self.padded_vocab_size + layer * self.padded_audio_vocab_size

def make_data_module(text_tokenizer: transformers.PreTrainedTokenizer, audio_processor, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = TA2TADataset(text_tokenizer=text_tokenizer, audio_processor=audio_processor, data_args=data_args)
    data_collator = train_dataset.collate_fn
    eval_dataset = TA2TADataset(text_tokenizer=text_tokenizer, audio_processor=audio_processor, data_args=data_args, split="eval")
    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=data_collator)
