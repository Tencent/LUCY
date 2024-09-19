import torch
import transformers
import einops
import soundfile as sf
from typing import Optional
from dataclasses import dataclass, field
from transformers import AutoTokenizer
from vita.model import VITAQwen2ForCausalLM, VITAQwen2Config
from transformers import AutoModelForCausalLM
from vita.util import s1_data_util
from vita.util.sampling import sample
from tqdm import tqdm
from vita.scripts.train_s1 import ModelArguments
from snac import SNAC

# ckpt_path = "outputs/vita_qwen2_s1-091124-130019/checkpoint-66000"
# ckpt_path = "outputs/vita_qwen2_s1_parallel-091324-121440/checkpoint-1000"
# model = VITAQwen2ForCausalLM.from_pretrained(ckpt_path)
# config = VITAQwen2Config.from_pretrained(ckpt_path)
# config.vocab_size = 181120
# config.tie_word_embeddings = False
# model = VITAQwen2ForCausalLM.from_pretrained(ckpt_path, config=config)
@dataclass
class InferenceArguments:
    ckpt_path: Optional[str] = field(default=None)
    max_code_length: Optional[int] = field(default=None)
    snac_sr: Optional[int] = field(default=24000)
    snac_model: Optional[str] = field(default="hubertsiuzdak/snac_24khz")
    output_path: Optional[str] = field(default=None)

def make_infer_inputs(item, dataset, device="cuda:0"):
    input_ids, input_length = dataset.extract_input([item]) # [L x T]
    input_ids = torch.Tensor(input_ids[0]).T.unsqueeze(0) # [L x T] => 1 x T x L
    attention_mask = torch.ones(input_ids.shape[:-1], dtype=bool)
    audio_features = item["audio"] # 1 x T x 3000
    use_audio_indices = torch.LongTensor([0]) if item["task"] == "ASR" else torch.LongTensor([])
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
    import pdb; pdb.set_trace()

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
    parser = transformers.HfArgumentParser((ModelArguments, s1_data_util.DataArguments, InferenceArguments))
    model_args, data_args, infer_args = parser.parse_args_into_dataclasses()
    s1_data_util.sync_data_args(model_args, data_args)
    print(model_args)
    print(data_args)
    print(infer_args)
    model = VITAQwen2ForCausalLM.from_pretrained(infer_args.ckpt_path).to(device)
    text_tokenizer = AutoTokenizer.from_pretrained(infer_args.ckpt_path)
    audio_processor = model.get_audio_encoder().audio_processor
    dataset = s1_data_util.ASRTTSDataset(text_tokenizer, audio_processor, data_args)
    snac = SNAC.from_pretrained(infer_args.snac_model, cache_dir=model_args.cache_dir).eval().to(device)
    T = infer_args.max_code_length # 12 hz code roughly equals to 30s
    audio_num_codebook = data_args.num_codebook
    audio_pads_shifted = torch.LongTensor([
        dataset.codec_layer_shift(dataset.PAD_A, i) 
        for i in range(audio_num_codebook)
    ]).to(device)
    text_pad = torch.LongTensor([dataset.PAD_T]).to(device)
    for i, item in enumerate(dataset):
        input_dict = make_infer_inputs(item, dataset, device)
        text_ends = False
        audio_ends = False
        audio_num_layer_ends = -1
        audio_tokens, text_tokens = [], []
        for t in tqdm(range(T)):
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
            next_input_ids = next_input_ids.view(1,1,-1)
            input_dict = {
                "input_ids": next_input_ids,
                "past_key_values": past_kv
            }
            current_text = text_tokenizer.decode(torch.cat(text_tokens)) 
            print(current_text)
        text_tokens = torch.cat(text_tokens)
        text = text_tokenizer.decode(text_tokens)
        audio = torch.stack(audio_tokens)
        wav = decode_audio(snac, audio).cpu().numpy().reshape(-1)
        sf.write(f'{infer_args.output_path}/{i}.wav', wav, infer_args.snac_sr)


if __name__ == "__main__":
    infer()

exit(0)

# print(model)
# cache_dir = "/mnt/data/hetinggao/models"
# text_tokenizer = AutoTokenizer.from_pretrained(
#     ckpt_path,
# )
# audio_processor = model.get_audio_encoder().audio_processor
# # data_args
# data_args = s1_data_util.DataArguments()

# text_additional = ["EOT", "PAD_T", "BOT", "ANS_T", "TTS"]
# audio_additional = ["EOA", "PAD_A", "BOA", "ANS_A", "ASR"]
# text_vocab_size = 151936
# text_vocab_size_padded = 152000
# audio_vocab_size = 4096
# audio_vocab_size_padded = 4160
# data_args_dict = {
#     "audio_manifest": "/mnt/data/hetinggao/manifest/sub1/train.tsv",
#     "transcript_file": "/mnt/data/hetinggao/manifest/sub1/train.wrd",
#     "codec_file": "/mnt/data/hetinggao/manifest/sub1/train.snac",
#     "text_additional_tokens": {token: text_vocab_size + i for i, token in enumerate(text_additional)},
#     "audio_additional_tokens":  {token: audio_vocab_size + i for i, token in enumerate(audio_additional)},
#     "padded_vocab_size": 152000,
#     "padded_audio_vocab_size": 4160
# }
# for k, v in data_args_dict.items():
#     setattr(data_args, k, v)

# dataset = s1_data_util.make_data_module(text_tokenizer, audio_processor, data_args)['train_dataset']
# print(data_args)

# item = dataset[0] # ASR
# input_ids, input_length = dataset.extract_input([item])
# input_ids = torch.Tensor(input_ids[0]).T.unsqueeze(0)
# attention_mask = torch.ones(input_ids.shape[:-1], dtype=bool)
# # import pdb; pdb.set_trace()
# # 
# audio_features = item['audio']
# use_audio_indices = torch.LongTensor([0])
# audio_lengths = torch.LongTensor([item["audio_length"]])
# # import pdb; pdb.set_trace()
# input_dict = {
#     "input_ids": input_ids.cuda(),
#     "attention_mask": attention_mask.cuda(),
#     "audios": audio_features.cuda(),
#     "audio_lengths": audio_lengths.cuda(),
#     "use_audio_indices": use_audio_indices.cuda(),
#     "output_labels_attention_mask": None,
#     "output_logits_attention_mask": None
# }
# model = model.cuda()




# @torch.inference_mode()
# def next_token(
#     model, 
#     audios=None,
#     attention_mask=None,
#     input_ids=None,
#     audio_lengths=None,
#     use_audio_indices=None,
#     past_key_values=None,
#     **kwargs,
# ) -> torch.Tensor:
#     # input_pos = input_pos.to(model.device)
#     # input_ids = [input_id.to(model.device) for input_id in input_ids]
#     outputs = model(
#         input_ids=input_ids, 
#         attention_mask=attention_mask,
#         audios=audios, 
#         audio_lengths=audio_lengths, 
#         use_audio_indices=use_audio_indices,
#         past_key_values=past_key_values,
#     )
#     next_t = sample(outputs.logits[...,:text_vocab_size_padded], top_k=1).to(input_ids[0])
#     next_a, next_a_unshifted = [], []

#     for i in range(data_args.num_codebook):
#         start = text_vocab_size_padded + i * audio_vocab_size_padded
#         end = text_vocab_size_padded + (i+1) * audio_vocab_size_padded
#         a_i = sample(outputs.logits[...,start:end], top_k=1)
#         next_a_i = dataset.codec_layer_shift(a_i, i)
#         next_a.append(next_a_i)
#         next_a_unshifted.append(a_i)
#     # next_t = sample(logit_t, **kwargs).to(dtype=input_ids[0].dtype)
    
#     next_a = torch.cat(next_a)

#     next_a_unshifted = torch.cat(next_a_unshifted)
#     past_key_values = outputs.past_key_values
#     return next_t, next_a, next_a_unshifted, past_key_values

# device = "cuda:0"
# cache_dir = "/mnt/data/hetinggao/mini-omni-main/hf_cache"
# sr = 24_000
# snac = SNAC.from_pretrained("hubertsiuzdak/snac_24khz", cache_dir=cache_dir).eval().to(device)
# text_tokens = []
# from time import time as ttime
# for ii in range(30):
#     T = 4 if ii > 1 else 7
#     t0 = ttime()
#     with torch.inference_mode():
#         # output = model(**input_dict)
#         audio_tokens = []
#         for i in tqdm(range(T)):
#             t, a, a_ori, past_key_values = next_token(model, **input_dict)
#             # import pdb; pdb.set_trace()
#             next_input_ids = torch.cat([a, t]).view(1,1,-1)
#             input_dict = {
#                 "input_ids": next_input_ids,
#                 "past_key_values": past_key_values
#             }
            
#             text_tokens.append(t)
#             # print(torch.cat([t, a_ori]).view(1,1,-1))
#             current_text = text_tokenizer.decode(torch.cat(text_tokens)) 
#             # print("Current:", current_text, torch.cat(text_tokens))
#             audio_tokens.append(a_ori)


#     audios = torch.stack(audio_tokens)
#     code_12hz, code_24hz, code_48hz = audios[:,0:1], audios[:,1:3], audios[:, 3:]
#     #audios_shifted = torch.zeros([T-data_args.num_codebook, data_args.num_codebook]).long().cuda() # T x 7

#     t1 = ttime()

#     #for i in range(data_args.num_codebook):
#     #    left_pad = i + 1
#     #    right_pad = data_args.num_codebook - left_pad
#     #    audios_shifted[:, i] = audios[left_pad:T-right_pad,i]
#     # code_12hz, code_24hz, code_48hz = audios_shifted[:,0:1], audios_shifted[:,1:3], audios_shifted[:, 3:]
#     code_12hz[code_12hz>=4096] = 4095
#     code_24hz[code_24hz>=4096] = 4095
#     code_48hz[code_48hz>=4096] = 4095
#     codes = [code_12hz.reshape(1, -1), code_24hz.reshape(1, -1), code_48hz.reshape(1,-1)]
#     t2 = ttime()
#     with torch.inference_mode():
#         audio = snac.decode(codes).view(-1)
#     t3 = ttime()
#     print(audio.shape, len(audio) / sr, 'generate', t1-t0, 'reshape', t2-t1, 'snac decode', t3-t2)


