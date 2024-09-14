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
    audio_manifest: Optional[str] = field(default=None)
    transcript_file: Optional[str] = field(default=None)
    codec_file: Optional[str] = field(default=None)
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
    tasks: Optional[List[str]] = field(default_factory=lambda: ["ASR", "TTS"])

def sync_data_args(model_args, data_args):
    setattr(data_args, "num_codebook", model_args.audio_num_codebook)    
    text_additional_tokens = {token: model_args.text_vocab_size + i for i, token in enumerate(model_args.text_additional)}
    audio_additional_tokens = {token: model_args.audio_vocab_size + i for i, token in enumerate(model_args.audio_additional)}
    setattr(data_args, "text_additional_tokens", text_additional_tokens)
    setattr(data_args, "audio_additional_tokens", audio_additional_tokens)
    setattr(data_args, "padded_vocab_size", model_args.text_vocab_size + model_args.text_special_tokens)
    setattr(data_args, "padded_audio_vocab_size", model_args.audio_vocab_size + model_args.audio_special_tokens)

# Modified from https://github.com/facebookresearch/fairseq/blob/main/fairseq/data/audio/hubert_dataset.py
def load_audio(manifest_path, transcript_path, max_keep, min_keep):
    n_long, n_short = 0, 0
    names, inds, sizes, trans = [], [], [], []
    with open(manifest_path) as f, open(transcript_path) as ftrans:
        root = f.readline().strip()
        for ind, (line, line_trans) in enumerate(zip(f, ftrans)):
            items = line.strip().split("\t")
            assert len(items) == 2, line
            sz = int(items[1])
            if min_keep is not None and sz < min_keep:
                n_short += 1
            elif max_keep is not None and sz > max_keep:
                n_long += 1
            else:
                names.append(items[0])
                inds.append(ind)
                sizes.append(sz)
                trans.append(line_trans.strip())
    tot = ind + 1
    logger.info(
        (
            f"max_keep={max_keep}, min_keep={min_keep}, "
            f"loaded {len(names)}, skipped {n_short} short and {n_long} long, "
            f"longest-loaded={max(sizes)}, shortest-loaded={min(sizes)}"
        )
    )
    return root, names, inds, tot, sizes, trans

# https://github.com/facebookresearch/fairseq/blob/main/fairseq/data/audio/hubert_dataset.py
def load_label_offset(label_path, inds, tot):
    with open(label_path) as f:
        code_lengths = [len(line.encode("utf-8")) for line in f]
        assert (
            len(code_lengths) == tot
        ), f"number of labels does not match ({len(code_lengths)} != {tot})"
        offsets = list(itertools.accumulate([0] + code_lengths))
        offsets = [(offsets[i], offsets[i + 1]) for i in inds]
    return offsets



class ASRTTSDataset(Dataset):
    
    def __init__(self, text_tokenizer: transformers.PreTrainedTokenizer, audio_processor: Any, data_args: DataArguments):
        super(ASRTTSDataset, self).__init__()
        self.manifest, self.transcripts_file, self.codec_file, self.audio_num_codebook = (
            data_args.audio_manifest, data_args.transcript_file, data_args.codec_file, data_args.num_codebook
        )
        (
            self.root_dir, 
            self.audio_paths, 
            self.ixs, 
            self.num_samples, 
            self.lengths,
            self.transcripts
        ) = load_audio(
            self.manifest, self.transcripts_file, 
            data_args.max_keep_sample_size, 
            data_args.min_keep_sample_size
        )

        self.codec_offsets_list = load_label_offset(self.codec_file, self.ixs, self.num_samples) 

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
        # audio_additional_tokens: { "EOA": 4096, "PAD_A": 4097, "BOA": 4098, "ANS_A": 4099, "ASR": 4100 } 
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

    def __len__(self):
        return self.num_samples * len(self.tasks)
    
    def __getitem__(self, index) -> Any:
        task = self.tasks[index // self.num_samples]
        ix = self.ix2ix[index%self.num_samples] # random shuffle if any
        audio_path = os.path.join(self.root_dir, self.audio_paths[ix])
        audio_length = self.lengths[ix]
        transcript = self.transcripts[ix]
        codec = self.get_codec(ix)
        wav, sr = sf.read(audio_path)
        if wav.ndim == 2:
            wav = wav.mean(-1)
        assert sr == self.sample_rate, f"Audio sampling rate {sr} != {self.sample_rate}"
        assert len(wav) == audio_length, f"Audio length {len(wav)} != {audio_length} of {audio_path}"
        audio = self.audio_processor(wav, sampling_rate=sr, return_tensors="pt").input_features 
        audio_length = int(audio_length / self.sample_rate * self.audio_feature_rate) + 1

        result = self.text_tokenizer([transcript], return_tensors="pt")
        text = result.input_ids.squeeze(0)

        data_dict = {
            "audio": audio,
            "audio_length": audio_length,
            "text": text,
            "codec": codec,
            "task": task
        }
        return data_dict

    def get_codec(self, index):
        with open(self.codec_file) as f:
            offset_s, offset_e = self.codec_offsets_list[index]
            f.seek(offset_s)
            codec = f.read(offset_e - offset_s)
        codec_array = list(map(int, codec.split()))
        return codec_array
    
    def extract_output(self, batch):
        outputs, output_lengths = [], []
        for item in batch:
            codec = torch.LongTensor(item["codec"])
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
                
            
            text = item["text"]
            text_length = len(text)
            assert text_length < codec_length_padded, \
                f"Number of audio codes {codec_length_padded}" \
                f"is not enough to cover texts of length {text_length}"
            num_text_pad = codec_length_padded - text_length - 1 # 1 for EOS token
            
            text_padded = torch.cat([
                text, torch.LongTensor([self.EOT]), 
                torch.zeros(num_text_pad, dtype=torch.long).fill_(self.PAD_T)]).unsqueeze(0) # 1 x T/7
            output = torch.cat([codec_by_layer_padded, text_padded], dim=0) # 8 x T/7
            outputs.append(output) # [8 x T / 7]

            output_lengs.append(text_length + 1)
            output_lengths.append(output_lengs) # B x 8
            assert output.shape[-1] == max(output_lengs)

        return outputs, output_lengths


    def extract_input(self, batch):
        inputs, input_lengths = [], []
        for item in batch:
            text, audio_length, task = item["text"], item["audio_length"], item["task"]
            
            if task == "ASR": # collate audio only
                codec_by_layer_padded = torch.zeros(
                    self.audio_num_codebook, audio_length+3, dtype=torch.long
                ) # 1 for BOA, 1 for EOA and 1 for ASR
                for i in range(self.audio_num_codebook):
                    codec_by_layer_padded[i] = torch.LongTensor(
                        [
                            self.codec_layer_shift(self.BOA, i)
                        ] + [
                            self.codec_layer_shift(self.PAD_A, i)
                        ] * audio_length + [
                            self.codec_layer_shift(self.EOA, i),
                            self.codec_layer_shift(self.ASR, i),
                        ]
                    )
                text_padded = torch.cat([
                    torch.LongTensor([self.BOT]),
                    torch.LongTensor([self.PAD_T] * audio_length),
                    torch.LongTensor([self.EOT, self.ANS_T])
                ]).unsqueeze(0) # 8 x (audio_length+3)
                input_length = audio_length + 3 # 1 for BOA, 1 for EOA and 1 for ASR
            elif task == "TTS": # collate text only
                codec_by_layer_padded = torch.zeros(
                    self.audio_num_codebook, len(text)+3, dtype=torch.long
                ) # 1 for BOT and 1 for EOT and 1 for ANS
                for i in range(self.audio_num_codebook):
                    codec_by_layer_padded[i] = torch.LongTensor(
                        [
                            self.codec_layer_shift(self.PAD_A, i)
                        ] * (len(text)+2) + [ # 1 for BOT and 1 for EOS
                            self.codec_layer_shift(self.ANS_A, i)
                        ]
                    )
                text_padded = torch.cat([
                    torch.LongTensor([self.BOT]), 
                    text, 
                    torch.LongTensor([self.EOT, self.TTS])]
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
        audio_lengths = torch.LongTensor([item["audio_length"] for item in batch if item["task"] == "ASR"])
        
        audios, use_audio_indices = None, None
        if len(audio_lengths) > 0:
            audios = torch.cat([item["audio"] for item in batch if item["task"] == "ASR"]) # B' x 80 x 3000
        else:

            audios = batch[0]["audio"].new(0,*batch[0]["audio"].shape[1:])
        use_audio_indices = torch.LongTensor([i for i, item in enumerate(batch) if item["task"] == "ASR"])

        input_ids, lengs = [], []
        
        for inp, inp_leng, outp, outp_lengs in zip(inputs, input_lengths, outputs, output_lengths):
            input_ids.append(torch.cat([inp, outp], dim=-1).T) # B x T x L
            lengs.append(inp_leng + max(outp_lengs)) # B x L
  
        input_ids, attention_mask = self.collate_sequence(
            input_ids, max(lengs), pad=0
        )

        output_labels_attention_mask = torch.zeros(input_ids.shape, dtype=bool) # B x T x L
        output_logits_attention_mask = torch.zeros(input_ids.shape, dtype=bool)
        for bi, (inp_leng, outp_lengs) in enumerate(zip(input_lengths, output_lengths)):
            for li, outp_leng in enumerate(outp_lengs):
                output_labels_attention_mask[bi, inp_leng:inp_leng+outp_leng, li] = True
                output_logits_attention_mask[bi, inp_leng-1:inp_leng+outp_leng-1, li] = True

        collated = {
            "input_ids": input_ids, 
            "attention_mask": attention_mask,
            "audios": audios,
            "audio_lengths": audio_lengths,
            "use_audio_indices": use_audio_indices,
            "output_labels_attention_mask": output_labels_attention_mask, # B x (T_in+T_out) x L
            "output_logits_attention_mask": output_logits_attention_mask # B x (T_in+T_out) x L

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
    train_dataset = ASRTTSDataset(text_tokenizer=text_tokenizer, audio_processor=audio_processor, data_args=data_args)
    data_collator = train_dataset.collate_fn
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)
