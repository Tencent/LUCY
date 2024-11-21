import torch
import transformers
import einops
import soundfile as sf
from typing import Optional
from dataclasses import dataclass, field
from transformers import AutoTokenizer
from vita.model import VITAQwen2ForCausalLM, VITAQwen2Config
from transformers import AutoModelForCausalLM
from vita.util import data_util
from vita.util.sampling import sample
from tqdm import tqdm
from vita.scripts.train import ModelArguments
from snac import SNAC
from time import time

@dataclass
class InferenceArguments:
    ckpt_path: Optional[str] = field(default=None)
    max_code_length: Optional[int] = field(default=None)
    snac_sr: Optional[int] = field(default=24000)
    snac_model: Optional[str] = field(default="hubertsiuzdak/snac_24khz")
    output_path: Optional[str] = field(default=None)
    save_audio: Optional[bool] = field(default=True)
    output_text_only: Optional[bool] = field(default=False)

def make_infer_inputs(item, dataset, device="cuda:0"):
    task = item["task"]
    input_ids, input_length = dataset.extract_input([item]) # [L x T]
    input_ids = torch.Tensor(input_ids[0]).T.unsqueeze(0) # [L x T] => 1 x T x L
    attention_mask = torch.ones(input_ids.shape[:-1], dtype=bool)
    audio_features = item["audio"] # 1 x T x 3000
    use_audio_indices = torch.LongTensor([0]) if task in dataset.tasks_with_audio_input else torch.LongTensor([])
    audio_lengths = torch.LongTensor([item["audio_length"]])
    input_dict = {
        "input_ids": input_ids.to(device),
        "attention_mask": attention_mask.to(device),
        "audios": audio_features.to(device),
        "audio_lengths": audio_lengths.to(device),
        "use_audio_indices": use_audio_indices.to(device),
        "output_labels_attention_mask": None,
        "output_logits_attention_mask": None
    }
    return input_dict

def next_token(
    model, 
    dataset,
    audios=None,
    attention_mask=None,
    input_ids=None,
    audio_lengths=None,
    use_audio_indices=None,
    past_key_values=None,
    **kwargs,
) -> torch.Tensor:
    data_args = dataset.data_args
    outputs = model(
        input_ids=input_ids, 
        attention_mask=attention_mask,
        audios=audios, 
        audio_lengths=audio_lengths, 
        use_audio_indices=use_audio_indices,
        past_key_values=past_key_values,
    )
    text_vocab_size_padded = data_args.padded_vocab_size
    audio_vocab_size_padded = data_args.padded_audio_vocab_size
    next_t = sample(outputs.logits[...,:text_vocab_size_padded], top_k=1).to(input_ids[0])
    next_a, next_ua = [], [] # layer shifted/unshifted audio tokens

    for i in range(data_args.num_codebook):
        start = text_vocab_size_padded + i * audio_vocab_size_padded
        end = text_vocab_size_padded + (i+1) * audio_vocab_size_padded
        ua_i = sample(outputs.logits[...,start:end], top_k=1)
        a_i = dataset.codec_layer_shift(ua_i, i)
        next_a.append(a_i)
        next_ua.append(ua_i)
    
    next_a = torch.cat(next_a)
    next_ua = torch.cat(next_ua)
    past_key_values = outputs.past_key_values
    return next_t, next_a, next_ua, past_key_values

def decode_audio(snac, audio_codes_padded):
    T, N = audio_codes_padded.shape # length of auido codes and number of codebooks
    audio_codes = torch.zeros((T-N-1, N)).to(audio_codes_padded) # 1 for EOA
    for i in range(N):
        audio_codes[:,i] = audio_codes_padded[i+1:-(N-i), i]
    (
        code_12hz, code_24hz, code_48hz
    ) = (
        audio_codes[:,0:1], 
        audio_codes[:,1:3],
        audio_codes[:,3:]
    )
    codes = [
        code_12hz.reshape(1, -1), 
        code_24hz.reshape(1, -1), 
        code_48hz.reshape(1, -1)
    ]
    audio = snac.decode(codes).view(-1)
    return audio


@torch.inference_mode()
def infer():
    device = "cuda:0"
    parser = transformers.HfArgumentParser((ModelArguments, data_util.DataArguments, InferenceArguments))
    model_args, data_args, infer_args = parser.parse_args_into_dataclasses()
    data_util.sync_data_args(model_args, data_args)
    print(model_args)
    print(data_args)
    print(infer_args)
    model = VITAQwen2ForCausalLM.from_pretrained(infer_args.ckpt_path).to(device)
    text_tokenizer = AutoTokenizer.from_pretrained(infer_args.ckpt_path)
    audio_processor = model.get_audio_encoder().audio_processor
    dataset = data_util.TA2TADataset(text_tokenizer, audio_processor, data_args)
    snac = SNAC.from_pretrained(infer_args.snac_model, cache_dir=model_args.cache_dir).eval().to(device)
    T = infer_args.max_code_length # 12 hz code roughly equals to 30s
    audio_num_codebook = data_args.num_codebook
    audio_pads_shifted = torch.LongTensor([
        dataset.codec_layer_shift(dataset.PAD_A, i) 
        for i in range(audio_num_codebook)
    ]).to(device)
    text_pad = torch.LongTensor([dataset.PAD_T]).to(device)
    with open(f"{infer_args.output_path}/hyp.txt", "w") as f:
        for i, item in enumerate(dataset):
            t0 = time()
            input_dict = make_infer_inputs(item, dataset, device)
            text_ends = False
            audio_ends = False
            audio_num_layer_ends = -1
            audio_tokens, text_tokens = [], []
            for t in tqdm(range(T)):
                if not infer_args.save_audio and text_ends:
                    break
                if audio_num_layer_ends == audio_num_codebook:
                    break
                next_t, next_a, next_ua, past_kv = next_token(
                    model, dataset, **input_dict
                )

                if t < audio_num_codebook:
                    num_pad = audio_num_codebook - t
                    next_a[-num_pad:] = audio_pads_shifted[-num_pad:]
                    next_ua[-num_pad:] = dataset.PAD_A
                if text_ends:
                    next_t = text_pad
                if audio_ends:
                    next_a[:audio_num_layer_ends] = audio_pads_shifted[:audio_num_layer_ends]
                    next_ua[:audio_num_layer_ends] = dataset.PAD_A
                    audio_num_layer_ends += 1
                audio_tokens.append(next_ua)
                text_tokens.append(next_t)
                if next_t == dataset.EOT:
                    text_ends = True
                if next_ua[0] == dataset.EOA:
                    audio_ends = True
                    audio_num_layer_ends = 1
                next_input_ids = torch.cat([next_a, next_t])
                if infer_args.output_text_only:
                    next_input_ids = torch.cat([audio_pads_shifted, next_t])
                next_input_ids = next_input_ids.view(1,1,-1)
                input_dict = {
                    "input_ids": next_input_ids,
                    "past_key_values": past_kv
                }
                current_text = text_tokenizer.decode(torch.cat(text_tokens)) 
                # print(current_text)
            text_tokens = torch.cat(text_tokens)
            text = text_tokenizer.decode(text_tokens)
            print(text.strip(), file=f)
            if infer_args.save_audio:
                audio = torch.stack(audio_tokens)
                wav = decode_audio(snac, audio).cpu().numpy().reshape(-1)
                sf.write(f'{infer_args.output_path}/{i}.wav', wav, infer_args.snac_sr)
            t1 = time()
            gen_time = t1 - t0
            wav_dur = len(wav) / infer_args.snac_sr
            print(f"Used {gen_time:.4f}s to generate {wav_dur:.4f}s audio with RTF: {gen_time/wav_dur}")

if __name__ == "__main__":
    infer()
