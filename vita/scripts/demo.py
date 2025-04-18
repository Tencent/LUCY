import torch
import transformers
import einops
import random
import numpy as np
import soundfile as sf
from typing import Optional
from dataclasses import dataclass, field
from transformers import AutoTokenizer
from vita.model.language_model.vita_qwen2_v2 import VITAQwen2ForCausalLM
from transformers import AutoModelForCausalLM
from tqdm import tqdm
from snac import SNAC
from time import time
from itertools import groupby
from vita.util import data_util_v2 as data_util
from vita.util.sampling import sample
from vita.scripts.train_v2 import ModelArguments
from vita.util import conversation as conversation_lib, move_to_cuda, move_to_cpu
from vita.constants import IGNORE_INDEX, AUDIO_PH, PAD_TOKEN

audio_num_codebook = 7
text_vocab_size_padded = 152000
audio_vocab_size_padded = 4160
EOA    = 4096
PAD_A  = 4097
BOA    = 4098
ANS_A  = 4099
PAD_T  = 151937
EOT    = 151936
IM_END = 151645

@dataclass
class InferenceArguments:
    max_code_length: Optional[int] = field(default=None)
    snac_sr: Optional[int] = field(default=24000)
    snac_model: Optional[str] = field(default="hubertsiuzdak/snac_24khz")
    output_path: Optional[str] = field(default=None)
    save_audio: Optional[bool] = field(default=True)
    output_text_only: Optional[bool] = field(default=False)

def next_token(
    model, 
    audios=None,
    attention_mask=None,
    input_ids=None,
    audio_lengths=None,
    audio_attention_mask=None,
    past_key_values=None,
    **kwargs,
) -> torch.Tensor:
    outputs = model(
        input_ids=input_ids, 
        attention_mask=attention_mask,
        audios=audios, 
        audio_lengths=audio_lengths, 
        audio_attention_mask=audio_attention_mask,
        past_key_values=past_key_values,
        use_cache=True
    )
    batch_size = input_ids.shape[0]
    assert batch_size == 1 or batch_size == 2, batch_size
    # if batch size is 2, use first item to predict audio codec and use second item to predict text
    logits_t = outputs.logits[-1:,:,:text_vocab_size_padded] # last item in batch
    # logits_t = outputs.logits[:1,:,:text_vocab_size_padded] # first item in batch
    next_t = sample(logits_t, top_k=1).to(input_ids[0]).repeat(batch_size).unsqueeze(-1) # B x 1

    next_a, next_ua = [], [] # layer shifted/unshifted audio tokens

    for i in range(audio_num_codebook):
        start = text_vocab_size_padded + i * audio_vocab_size_padded
        end = text_vocab_size_padded + (i+1) * audio_vocab_size_padded
        logits_a_i = outputs.logits[:1, :,start:end]
        ua_i = input_ids.new_zeros(batch_size,1).fill_(PAD_A)
        ua_i[:1, :] = sample(logits_a_i, top_k=10) # B x 1 # first item in batch
        a_i = codec_layer_shift(ua_i, i) # B x 1
        next_a.append(a_i)
        next_ua.append(ua_i)
    
    next_a = torch.cat(next_a, dim=-1) # B x 7
    next_ua = torch.cat(next_ua, dim=-1) # B x 7
    past_key_values = outputs.past_key_values
    return next_t, next_a, next_ua, past_key_values

def decode_audio(snac, audio_codes_padded):
    T, N = audio_codes_padded.shape # length of auido codes and number of codebooks
    audio_codes = torch.zeros((T-N-1, N)).to(audio_codes_padded) # 1 for EOA
    for i in range(N):
        audio_codes[:,i] = audio_codes_padded[i+1:-(N-i), i]
    print(audio_codes)
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

def load_wav_feat(wavpaths, audio_processor, sample_rate=16_000, audio_feature_rate=50):
    wav, sr = load_wav(wavpaths)
    assert sr == sample_rate, f"{sr} != {sample_rate}"
    audio = audio_processor(wav, sampling_rate=sr, return_tensors="pt").input_features
    audio_length = len(wav)
    audio_length = int(audio_length / sample_rate * audio_feature_rate) + 1
    return audio, audio_length

def codec_layer_shift(input_id, layer):
    return input_id + text_vocab_size_padded + layer * audio_vocab_size_padded



def prepare_inputs(
        source, use_audio_input, tokenizer, audio_processor, add_system_prompt, 
        past_input_dict=None, generated=None
    ):
    shifted_PAD_A = torch.LongTensor([codec_layer_shift(PAD_A, i) for i in range(audio_num_codebook)])
    shifted_BOA   = torch.LongTensor([codec_layer_shift(BOA, i)   for i in range(audio_num_codebook)])
    shifted_EOA   = torch.LongTensor([codec_layer_shift(EOA, i)   for i in range(audio_num_codebook)])
    shifted_ANS_A = torch.LongTensor([codec_layer_shift(ANS_A, i) for i in range(audio_num_codebook)])
    AUDIO_PH_idx  = tokenizer.convert_tokens_to_ids(AUDIO_PH)
    PAD_TOKEN_idx = tokenizer.convert_tokens_to_ids(PAD_TOKEN)
    conv = conversation_lib.conv_qwen2.copy()
    conv.messages = []

    audios, audio_lengths = torch.zeros([0,80,3000]), torch.zeros([0]).long()
    # import pdb; pdb.set_trace()
    if past_input_dict is not None:
        audio_lengths = past_input_dict["audio_lengths"]
        num_audio = len(audio_lengths) // 2
        audio_lengths = audio_lengths[:num_audio]
        audios = past_input_dict["audios"][:num_audio]
    has_audio_input = "wavpath" in source

    if has_audio_input and use_audio_input:
        audio, audio_length = load_wav_feat(source["wavpath"], audio_processor)
        message = AUDIO_PH * (audio_length + 2)
        # audios.append(audio)
        audios = torch.cat([audios, audio])
        audio_lengths = torch.cat([
            audio_lengths, torch.LongTensor([audio_length])
        ])

    else:
        message = source["content"]
    role = source["role"]
    if add_system_prompt:
        conv.append_message(role, message)
        prompt = conv.get_prompt()
    else:
        prompt = f"<|im_start|>user\n{message}<|im_end|>\n"
    prompt += "<|im_start|>assistant\n"
    
    input_ids = tokenizer.encode(prompt, return_tensors="pt")[0]
    if past_input_dict is not None:
        input_ids = torch.cat([
            past_input_dict["input_ids"][0,:,-1], generated, input_ids]
        )

    input_codec = input_ids.new_zeros([len(input_ids), audio_num_codebook]).fill_(IGNORE_INDEX) # T x 7
    input_codec[:,:] = shifted_PAD_A[None,:]

    i_chunk, start, end = 0, 0, 0
    audio_attention_mask = input_ids == AUDIO_PH_idx
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
    input_ids = torch.cat([input_codec, input_ids.unsqueeze(-1)], dim=-1) # T x 8
    batched_input_ids = input_ids.unsqueeze(0).repeat(2, 1, 1) # 2 x T x 8
    batched_input_ids[0, -1, :-1] = shifted_ANS_A # the last position of the first item in the batch is ANS_A
    batched_audio_attention_mask = audio_attention_mask.unsqueeze(0).expand(2, -1) # 2 x T
    audio_lengths = audio_lengths.repeat(2) 
    attention_mask = batched_input_ids[...,-1].ne(PAD_TOKEN_idx)
    assert attention_mask.all()
    
    audios = torch.cat([audios, audios]) 
    input_dict = {
        "input_ids": batched_input_ids,
        "labels": None,
        "attention_mask": attention_mask, 
        "audios": audios,
        "audio_lengths": audio_lengths,
        "audio_attention_mask": batched_audio_attention_mask
    }
    return input_dict

def get_past_kv(past_kv, index):
    past_kv_i = tuple([tuple([x[index:index+1] for x in l]) for l in past_kv])
    return past_kv_i

def repeat_past_kv(past_kv, n):
    past_kv_n = tuple([tuple([x.repeat(n,1,1,1) for x in l]) for l in past_kv])
    return past_kv_n
    

def batch_parallel_decode(model, tokenizer, input_dict, infer_args, device):
    audio_pads_shifted = torch.LongTensor([
        codec_layer_shift(PAD_A, i) for i in range(audio_num_codebook)
    ]).to(device)
    text_pad = torch.LongTensor([PAD_T]).to(device)
    text_ends = False
    audio_ends = False
    audio_num_layer_ends = -1
    audio_tokens, text_tokens = [], []
    for t in tqdm(range(infer_args.max_code_length)):
        if not infer_args.save_audio and text_ends:
            break
        if audio_num_layer_ends == audio_num_codebook:
            break
        next_t, next_a, next_ua, past_kv = next_token(
            model, **input_dict
        )
        # past_kv (num_layer x (2 x [B, 2, T, 128]) )

        if t < audio_num_codebook:
            num_pad = audio_num_codebook - t
            next_a[0,-num_pad:] = audio_pads_shifted[-num_pad:]
            next_ua[0,-num_pad:] = PAD_A
        if text_ends:
            next_t[0] = text_pad
        if audio_ends:
            next_a[0,:audio_num_layer_ends] = audio_pads_shifted[:audio_num_layer_ends]
            next_ua[0,:audio_num_layer_ends] = PAD_A
            audio_num_layer_ends += 1
        audio_tokens.append(next_ua[0])
        if len(text_tokens) > 0 and text_tokens[-1] == IM_END:
            next_t[:] = EOT
            # save past_key_values of second item and retain only first item in the batch
            next_t = next_t[:1]
            next_a = next_a[:1]
            # past_kv_t = get_past_kv(past_kv, 1)
            past_kv = get_past_kv(past_kv, 0)

        text_tokens.append(next_t[0])

        if next_t[0] == EOT:
            text_ends = True
        if next_ua[0,0] == EOA:
            audio_ends = True
            audio_num_layer_ends = 1
        next_input_ids = torch.cat([next_a, next_t], dim=-1)
        batch_size = next_input_ids.shape[0]
        if infer_args.output_text_only:
            next_input_ids = torch.cat([audio_pads_shifted.unsqueeze(0).repeat(batch_size, -1), next_t], dim=-1)
        next_input_ids = next_input_ids.view(batch_size,1,audio_num_codebook+1)
        input_dict = {
            "input_ids": next_input_ids,
            "past_key_values": past_kv
        }
        if not text_ends:
            current_text = tokenizer.decode(torch.cat(text_tokens)) 
            print(current_text)
    text_tokens = torch.cat(text_tokens)
    audio_tokens = torch.stack(audio_tokens)
    return text_tokens, audio_tokens
    
@torch.inference_mode()
def demo(use_audio_input=False):
    
    print("use_audio_input", use_audio_input)
    parser = transformers.HfArgumentParser((ModelArguments, data_util.DataArguments, InferenceArguments))
    model_args, data_args, infer_args = parser.parse_args_into_dataclasses()
    data_util.sync_data_args(model_args, data_args)
    print(model_args)
    print(data_args)
    print(infer_args)
    device = torch.cuda.current_device()
    model = VITAQwen2ForCausalLM.from_pretrained(model_args.model_name_or_path).eval().to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    audio_processor = model.get_audio_encoder().audio_processor
    snac = SNAC.from_pretrained(infer_args.snac_model, cache_dir=model_args.cache_dir).eval().to(device)

    texts = ""
    # past_kv_t = None
    past_input_dict, generated = None, None
    for i, source in enumerate(conversation):
        if source["role"] != "user":
            continue
        add_system_prompt = i == 0
        t0 = time()
        input_dict = prepare_inputs(source, use_audio_input, tokenizer, audio_processor, add_system_prompt, past_input_dict, generated)
        # if past_kv_t is not None:
        #    input_dict["past_key_values"] = repeat_past_kv(past_kv_t, 2)
        input_dict = move_to_cuda(input_dict, device)
        text_tokens, audio_tokens = batch_parallel_decode(model, tokenizer, input_dict, infer_args, device)
        

        if infer_args.save_audio:
            wav = decode_audio(snac, audio_tokens).cpu().numpy().reshape(-1)
            sf.write(f'{infer_args.output_path}/{i}.wav', wav, infer_args.snac_sr)
        t1 = time()
        gen_time = t1 - t0
        wav_dur = len(wav) / infer_args.snac_sr
        print(f"Used {gen_time:.4f}s to generate {wav_dur:.4f}s audio with RTF: {gen_time/wav_dur}")

        text = tokenizer.decode(text_tokens)
        texts += f"{text.strip()}\n\n\n"

        past_input_dict = move_to_cpu(input_dict)
        generated = text_tokens[(text_tokens!=PAD_T)&(text_tokens!=EOT)].cpu()
        break

    with open(f"{infer_args.output_path}/hyp.txt", "w") as f:
        f.write(texts)
            
conversation = [{
    'content': '你知道“习近平新时代中国特色社会主义思想”这门课程吗', 
    'language': 'Chinese', 
    'redacted': False, 
    'role': 'user', 
    'toxic': False, 
    'wavpath': '/mnt/data/alanhshao/vita-e2e/CosyVoice/vita2tts/generated/WildChat/d180955b72fbb588877cacd07e54bbbe_turn0_q.wav', 
    'codec': '/mnt/data/alanhshao/vita-e2e/CosyVoice/vita2tts/generated/WildChat/d180955b72fbb588877cacd07e54bbbe_turn0_q.snac'
}, {
    'content': '我是人工智能语言模型，知道习近平新时代中国特色社会主义思想这门课程，它是习近平总书记在2017年10月18日中国共产党第十九次全国代表大会'
    '上作报告时提出的，是新时代中国特色社会主义的理论体系，也是中国共产党领导下中国特色社会主义事业的总纲领。该课程对建设中国特色社会主义的历史逻辑、'
    '发展路径、目标任务、理论基础和实践保障等方面作了系统深入的阐述，是当前国内高校中开设的重要课程之一。', 
    'language': 'Chinese', 
    'redacted': False, 
    'role': 'assistant', 
    'toxic': False, 
    'wavpath': '/mnt/data/alanhshao/vita-e2e/CosyVoice/vita2tts/generated/WildChat/d180955b72fbb588877cacd07e54bbbe_turn0_a.wav', 
    'codec': '/mnt/data/alanhshao/vita-e2e/CosyVoice/vita2tts/generated/WildChat/d180955b72fbb588877cacd07e54bbbe_turn0_a.snac'
}, {
    'content': '现在，假设你是一名学习这门课程的学生，请你用1000字的篇幅来表达你学习完这门课程后的收获与体会', 
    'language': 'Chinese',
    'redacted': False, 
    'role': 'user', 
    'toxic': False, 
    'wavpath': '/mnt/data/alanhshao/vita-e2e/CosyVoice/vita2tts/generated/WildChat/d180955b72fbb588877cacd07e54bbbe_turn1_q.wav', 
    'codec': '/mnt/data/alanhshao/vita-e2e/CosyVoice/vita2tts/generated/WildChat/d180955b72fbb588877cacd07e54bbbe_turn1_q.snac'
}, {
    'content': '作为一名学习“习近平新时代中国特色社会主义思想”这门课程的学生，我深受其启发并受益匪浅。通过学习这门课程，我对中国特色社会主义'
    '的理论体系及其实践要求有了更深入的了解，同时也对我作为一名公民应该承担的责任和任务有了更为明确的认识。\n\n首先，学习“习近平新时代中国特色'
    '社会主义思想”让我深刻认识到了中国特色社会主义道路的深刻意义。与西方国家不同，中国特色社会主义这一道路是经过试验和实践的，是适合中国国情的，'
    '具有可持续性和实现全面发展的优势。这样的道路不仅保障了国家的稳定，而且让中国特色社会主义在国际关系中扮演了更为重要的角色。\n\n其次，这门课'
    '程让我更加深入了解了共产党员的责任和使命。我们要认识到，共产党员不仅是一个党派成员，更是一名为人民服务的公民。我们要时刻以人民为中心，积极'
    '传递中国特色社会主义道路的优势，促进中国在国际舞台上的崛起。\n\n此外，这门课程通过案例分析来探讨如何应对国内和国际的挑战。中国今天面临的局'
    '面是前所未有的。我们不仅要维护国家的安全和稳定，同时要推进经济发展、加强基础设施建设、保障社会和谐稳定，树立中国作为一个大国的形象。本课程'
    '通过深入地探讨了这些问题，并介绍了可以应对挑战的策略。这对我们今后工作中的思考非常有帮助。\n\n总之，“习近平新时代中国特色社会主义思想”这门'
    '课程对我产生了深刻的影响。它让我更好地了解了共产党的发展历史并深入了解了中国特色社会主义的现实意义。在今后我的职业发展中，这门课程为我提供了'
    '很好的指导和参考价值。', 
    'language': 'Chinese', 
    'redacted': False, 
    'role': 'assistant', 
    'toxic': False, 
    'wavpath': '/mnt/data/alanhshao/vita-e2e/CosyVoice/vita2tts/generated/WildChat/d180955b72fbb588877cacd07e54bbbe_turn1_a.wav', 
    'codec': '/mnt/data/alanhshao/vita-e2e/CosyVoice/vita2tts/generated/WildChat/d180955b72fbb588877cacd07e54bbbe_turn1_a.snac'
}, {
    'content': '请你更加详细地论述一遍，不少于1000字', 
    'language': 'Chinese', 
    'redacted': False, 
    'role': 'user', 
    'toxic': False, 
    'wavpath': '/mnt/data/alanhshao/vita-e2e/CosyVoice/vita2tts/generated/WildChat/d180955b72fbb588877cacd07e54bbbe_turn2_q.wav',
    'codec': '/mnt/data/alanhshao/vita-e2e/CosyVoice/vita2tts/generated/WildChat/d180955b72fbb588877cacd07e54bbbe_turn2_q.snac'
}, {
    'content': '作为一名学习“习近平新时代中国特色社会主义思想”这门课程的学生，我认为，这门课程对于理解和认识中国特色社会主义的理论体系和实践'
    '有着非常重要的作用。\n\n首先，这门课程让我更加深刻地认识到了中国特色社会主义的深刻意义。在历史的变革和世界的发展背景下，中国选择了自己的'
    '路，探索了特色社会主义道路。这样的的路，不仅保障了国家的稳定和发展，而且使得中国特色社会主义在世界上的地位得到了提高。更为重要的是，中国'
    '特色社会主义的一系列探索给世界提供了宝贵的经验和借鉴，成为人民群众推进建设美好社会的成功方法。因此，我们每一个公民都需要深入地认识和了解'
    '中国特色社会主义的实践成果和理论路线，将这些理论路线落地到具体的实践中.\n\n\n其次，这门课程帮助我认识到作为一名共产党员的责任和使命。我'
    '们要时刻以人民为中心，将人民的利益放在第一位，忠实履行我们作为党员的职责所在。我们要积极传递中国特色社会主义道路的优势，以促进中国在国际'
    '舞台上的崛起。同时，我们也要在生产、学习和工作中树立正确的世界观、价值观和人生观，以更好地为实现构建美好社会而努力。因此，作为一名共产党'
    '员，我们既要把握人民根本利益，积极为人民服务，也要努力提高自身素质和对理论体系的理解和领悟。\n\n此外，这门课程通过案例分析探讨如何应对国'
    '内和国际的挑战。在当前国际形势非常严峻的情况下，中国在发展过程中面临着数个难以克服的困难和挑战。对于这些困难和挑战，本课程提出了一系列的'
    '解决策略和方案。通过深入地学习和了解这些策略和方案，我们可以更好地应对挑战和困难。本课程使我对中国特色社会主义的道路和发展战略有了更为全'
    '面和深入的了解，对我自己的职业生涯和工作过程中的决策和做事方法提供了有力指导。\n\n总之，“习近平新时代中国特色社会主义思想”这门课程对我影'
    '响很大。它使我更加全面深刻地认识到了中国特色社会主义的理论体系和实践成果，让我有了更加清晰的认识和立场。在今后的工作中，我将继续学习和宣'
    '传中国特色社会主义道路，积极为人民服务，并为推进中国在国际舞台上的崛起不懈努力。', 
    'language': 'Chinese', 
    'redacted': False, 
    'role': 'assistant', 
    'toxic': False,
    'wavpath': '/mnt/data/alanhshao/vita-e2e/CosyVoice/vita2tts/generated/WildChat/d180955b72fbb588877cacd07e54bbbe_turn2_a.wav', 
    'codec': '/mnt/data/alanhshao/vita-e2e/CosyVoice/vita2tts/generated/WildChat/d180955b72fbb588877cacd07e54bbbe_turn2_a.snac'
}]

conversation = [{'role': 'user', 'content': '你好'}, {'role': 'assistant', 'content': '你好！有什么我可以帮助你的吗？', 'wavpath': '/mnt/data/alanhshao/vita-e2e/CosyVoice/vita2tts/generated/AudioAgent/fba0222a9d017932c74f640eef0954ef_turn0_a.wav', 'codec': '/mnt/data/alanhshao/vita-e2e/CosyVoice/vita2tts/generated/AudioAgent/fba0222a9d017932c74f640eef0954ef_turn0_a.snac'}, {'role': 'user', 'content': '中孚信息是干什么的', 'wavpath': '/mnt/data/alanhshao/vita-e2e/CosyVoice/vita2tts/generated/AudioAgent/fba0222a9d017932c74f640eef0954ef_turn1_q.wav', 'codec': '/mnt/data/alanhshao/vita-e2e/CosyVoice/vita2tts/generated/AudioAgent/fba0222a9d017932c74f640eef0954ef_turn1_q.snac'}, {'role': 'assistant', 'content': '中孚信息是一家以信息技术服务为主导的公司。作为中国大型的IT服务提供商之一，中孚信息为客户提供全方位的信息技术解决方案，包括软件开发、系统集成、云计算、大数据分析、人工智能等领域。中孚信息致力于为企业客户提供高质量的技术支持和创新解决方案，帮助他们提升业务效率和竞争力。', 'wavpath': '/mnt/data/alanhshao/vita-e2e/CosyVoice/vita2tts/generated/AudioAgent/fba0222a9d017932c74f640eef0954ef_turn1_a.wav', 'codec': '/mnt/data/alanhshao/vita-e2e/CosyVoice/vita2tts/generated/AudioAgent/fba0222a9d017932c74f640eef0954ef_turn1_a.snac'}, {'role': 'user', 'content': '中孚信息是国企吗', 'wavpath': '/mnt/data/alanhshao/vita-e2e/CosyVoice/vita2tts/generated/AudioAgent/fba0222a9d017932c74f640eef0954ef_turn2_q.wav', 'codec': '/mnt/data/alanhshao/vita-e2e/CosyVoice/vita2tts/generated/AudioAgent/fba0222a9d017932c74f640eef0954ef_turn2_q.snac'}, {'role': 'assistant', 'content': '是的，中孚信息是一家国有企业。它成立于1999年，总部位于中国北京，并在全国范围内拥有多家分支机构。作为国有企业，中孚信息在信息技术领域扮演着重要的角色，为国家和社会提供技术支持和解决方案。', 'wavpath': '/mnt/data/alanhshao/vita-e2e/CosyVoice/vita2tts/generated/AudioAgent/fba0222a9d017932c74f640eef0954ef_turn2_a.wav', 'codec': '/mnt/data/alanhshao/vita-e2e/CosyVoice/vita2tts/generated/AudioAgent/fba0222a9d017932c74f640eef0954ef_turn2_a.snac'}, {'role': 'user', 'content': '不是国企', 'wavpath': '/mnt/data/alanhshao/vita-e2e/CosyVoice/vita2tts/generated/AudioAgent/fba0222a9d017932c74f640eef0954ef_turn3_q.wav', 'codec': '/mnt/data/alanhshao/vita-e2e/CosyVoice/vita2tts/generated/AudioAgent/fba0222a9d017932c74f640eef0954ef_turn3_q.snac'}, {'role': 'assistant', 'content': '抱歉，我之前给出的回答有误。根据最新信息，中孚信息是一家民营企业，而非国有企业。非常抱歉给您带来了困扰。谢谢您的指正。', 'wavpath': '/mnt/data/alanhshao/vita-e2e/CosyVoice/vita2tts/generated/AudioAgent/fba0222a9d017932c74f640eef0954ef_turn3_a.wav', 'codec': '/mnt/data/alanhshao/vita-e2e/CosyVoice/vita2tts/generated/AudioAgent/fba0222a9d017932c74f640eef0954ef_turn3_a.snac'}]

conversation = [{'content': '一名英国大学本科毕业生可以在雅思考试中取得多少分？', 'language': 'Chinese', 'redacted': False, 'role': 'user', 'toxic': False, 'wavpath': '/mnt/data/alanhshao/vita-e2e/CosyVoice/vita2tts/generated/WildChat/1170888a238b0890fe025e2499c4b7c4_turn0_q.wav', 'codec': '/mnt/data/alanhshao/vita-e2e/CosyVoice/vita2tts/generated/WildChat/1170888a238b0890fe025e2499c4b7c4_turn0_q.snac'}, {'content': '这个问题很难回答，因为雅思考试的分数是由多个因素决定的，而一个人是否是英国大学本科毕业生只是其中之一。根据雅思官方网站，考试分数的因素包括听力、阅读、写作和口语四个部分，每个部分的分数范围都是0-9分，总分最高为9分。因此，一个英国大学本科毕业生的雅思考试得分将取决于他们的英语水平，而不是仅仅因为他们完成了本科教育。', 'language': 'Chinese', 'redacted': False, 'role': 'assistant', 'toxic': False, 'wavpath': '/mnt/data/alanhshao/vita-e2e/CosyVoice/vita2tts/generated/WildChat/1170888a238b0890fe025e2499c4b7c4_turn0_a.wav', 'codec': '/mnt/data/alanhshao/vita-e2e/CosyVoice/vita2tts/generated/WildChat/1170888a238b0890fe025e2499c4b7c4_turn0_a.snac'}, {'content': '一位英国人参加雅思考试能达到什么水平？', 'language': 'Chinese', 'redacted': False, 'role': 'user', 'toxic': False, 'wavpath': '/mnt/data/alanhshao/vita-e2e/CosyVoice/vita2tts/generated/WildChat/1170888a238b0890fe025e2499c4b7c4_turn1_q.wav', 'codec': '/mnt/data/alanhshao/vita-e2e/CosyVoice/vita2tts/generated/WildChat/1170888a238b0890fe025e2499c4b7c4_turn1_q.snac'}, {'content': '英国人参加雅思考试的水平因人而异，无法一概而论。雅思考试主要测试听力、阅读、写作和口语四个方面的英语能力，考生的得分取决于多个因素，包括英语水平、备考时间、考前准备等等。根据雅思考试官方网站的数据，英国考生的雅思总分平均为6.5分，但是这只是一个平均值，而且得分在考试中的分布非常广泛。有些英国考生可以获得近乎完美的分数，而有些人的得分则比较低。', 'language': 'Chinese', 'redacted': False, 'role': 'assistant', 'toxic': False, 'wavpath': '/mnt/data/alanhshao/vita-e2e/CosyVoice/vita2tts/generated/WildChat/1170888a238b0890fe025e2499c4b7c4_turn1_a.wav', 'codec': '/mnt/data/alanhshao/vita-e2e/CosyVoice/vita2tts/generated/WildChat/1170888a238b0890fe025e2499c4b7c4_turn1_a.snac'}, {'content': '平均分为多少？', 'language': 'Chinese', 'redacted': False, 'role': 'user', 'toxic': False, 'wavpath': '/mnt/data/alanhshao/vita-e2e/CosyVoice/vita2tts/generated/WildChat/1170888a238b0890fe025e2499c4b7c4_turn2_q.wav', 'codec': '/mnt/data/alanhshao/vita-e2e/CosyVoice/vita2tts/generated/WildChat/1170888a238b0890fe025e2499c4b7c4_turn2_q.snac'}, {'content': '根据雅思官方网站提供的数据，截止到2021年6月，全球各地区考生雅思总分的平均分数如下：\n\n- 全球平均分：约6.0分\n- 欧洲地区平均分：约6.2分\n- 中国大陆地区平均分：约5.4分\n- 英国考生平均分：约6.5分\n\n当然，这些数据只是参考值，具体得分还要看每个考生自身的英语水平和考试情况。', 'language': 'Chinese', 'redacted': False, 'role': 'assistant', 'toxic': False, 'wavpath': '/mnt/data/alanhshao/vita-e2e/CosyVoice/vita2tts/generated/WildChat/1170888a238b0890fe025e2499c4b7c4_turn2_a.wav', 'codec': '/mnt/data/alanhshao/vita-e2e/CosyVoice/vita2tts/generated/WildChat/1170888a238b0890fe025e2499c4b7c4_turn2_a.snac'}]

conversation = [{'role': 'user', 'content': '中孚信息是干什么的', 'wavpath': '/mnt/data/alanhshao/vita-e2e/CosyVoice/vita2tts/generated/AudioAgent/fba0222a9d017932c74f640eef0954ef_turn1_q.wav', 'codec': '/mnt/data/alanhshao/vita-e2e/CosyVoice/vita2tts/generated/AudioAgent/fba0222a9d017932c74f640eef0954ef_turn1_q.snac'}, {'role': 'assistant', 'content': '中孚信息是一家以信息技术服务为主导的公司。作为中国大型的IT服务提供商之一，中孚信息为客户提供全方位的信息技术解决方案，包括软件开发、系统集成、云计算、大数据分析、人工智能等领域。中孚信息致力于为企业客户提供高质量的技术支持和创新解决方案，帮助他们提升业务效率和竞争力。', 'wavpath': '/mnt/data/alanhshao/vita-e2e/CosyVoice/vita2tts/generated/AudioAgent/fba0222a9d017932c74f640eef0954ef_turn1_a.wav', 'codec': '/mnt/data/alanhshao/vita-e2e/CosyVoice/vita2tts/generated/AudioAgent/fba0222a9d017932c74f640eef0954ef_turn1_a.snac'}, {'role': 'user', 'content': '中孚信息是国企吗', 'wavpath': '/mnt/data/alanhshao/vita-e2e/CosyVoice/vita2tts/generated/AudioAgent/fba0222a9d017932c74f640eef0954ef_turn2_q.wav', 'codec': '/mnt/data/alanhshao/vita-e2e/CosyVoice/vita2tts/generated/AudioAgent/fba0222a9d017932c74f640eef0954ef_turn2_q.snac'}, {'role': 'assistant', 'content': '是的，中孚信息是一家国有企业。它成立于1999年，总部位于中国北京，并在全国范围内拥有多家分支机构。作为国有企业，中孚信息在信息技术领域扮演着重要的角色，为国家和社会提供技术支持和解决方案。', 'wavpath': '/mnt/data/alanhshao/vita-e2e/CosyVoice/vita2tts/generated/AudioAgent/fba0222a9d017932c74f640eef0954ef_turn2_a.wav', 'codec': '/mnt/data/alanhshao/vita-e2e/CosyVoice/vita2tts/generated/AudioAgent/fba0222a9d017932c74f640eef0954ef_turn2_a.snac'}, {'role': 'user', 'content': '不是国企', 'wavpath': '/mnt/data/alanhshao/vita-e2e/CosyVoice/vita2tts/generated/AudioAgent/fba0222a9d017932c74f640eef0954ef_turn3_q.wav', 'codec': '/mnt/data/alanhshao/vita-e2e/CosyVoice/vita2tts/generated/AudioAgent/fba0222a9d017932c74f640eef0954ef_turn3_q.snac'}, {'role': 'assistant', 'content': '抱歉，我之前给出的回答有误。根据最新信息，中孚信息是一家民营企业，而非国有企业。非常抱歉给您带来了困扰。谢谢您的指正。', 'wavpath': '/mnt/data/alanhshao/vita-e2e/CosyVoice/vita2tts/generated/AudioAgent/fba0222a9d017932c74f640eef0954ef_turn3_a.wav', 'codec': '/mnt/data/alanhshao/vita-e2e/CosyVoice/vita2tts/generated/AudioAgent/fba0222a9d017932c74f640eef0954ef_turn3_a.snac'}]

# conversation = [{'role': 'user', 'content': '题目：假设小明手上有5颗水果硬糖，他将其中的3颗糖果分给了小红，还余下2颗糖果。这段情节是真实的还是虚构的？', "wavpath": "/mnt/data/alanhshao/vita-e2e/CosyVoice/vita2tts/generated/Infinity0729_50K/69f3c38c6b1033cf4c0a3e24d4a3e3d0_turn0_q.wav"}]

# conversation = [{'role': 'user', 'content': 'xcos2Πy+ysin2Πx在x,y,[-2, 2]、[-2, 2]的范围内的极大值，x,y分别取何值保留4位小数，只给答案不给过程', "wavpath": "/mnt/data/alanhshao/vita-e2e/CosyVoice/vita2tts/generated/WildChat/502613b9ed4f13ed6611dcda1953bf75_turn0_q.wav"}]

# conversation = [{'role': 'user', 'content': '王鹏家今天第一季度共交了水费138元，三月份交水费48元，王鹏家平均每月交多少元水费？', "wavpath": "/mnt/data/alanhshao/vita-e2e/CosyVoice/vita2tts/generated/KCG60w_50K/db1aef1af18fe9070c6bcd4eb94be2c1_turn0_q.wav"}]

# conversation = [{'role': 'user', 'content': '将一个棱长为6的正方体切割成若干个相同的棱长为整数的小正方体，若这些小正方体的表面积之和是切割前的大正方体的表面积的2倍，则切割成的小正方体的棱长是____。', "wavpath": "/mnt/data/alanhshao/vita-e2e/CosyVoice/vita2tts/generated/KCG60w_50K/915d183abc3ffab94dd63eb56bac3193_turn0_q.wav"}]

# conversation = [{'role': 'user', 'content': '$25x25=$'}]
# conversation = [{'role': 'user', 'content': 'The value of $\\log_{8}{2938}$ is between the consecutive integers $a$ and $b$.  Find $a+b$.'}]
# conversation = [{'role': 'user', 'content': '介绍一下中国首都', "wavpath": "/mnt/data/hetinggao/vita-e2e/downloads/tmp8v4j2hiy_0_2.5.wav"}]
# conversation = [{'role': 'user', 'content': '请问你是谁', "wavpath": "/mnt/data/hetinggao/vita-e2e/downloads/tmpo19y4f3g.wav"}]
# conversation = [{'role': 'user', 'content': '请问你是谁', "wavpath": "/mnt/data/hetinggao/vita-e2e/downloads/tmpxyopwt87.wav"}]
# conversation = [{'role': 'user', 'content': '请问你是谁', "wavpath": "/mnt/data/hetinggao/vita-e2e/downloads/tmp7m_hpts4.wav"}]
conversation = [{'role': 'user', 'content': '讲一个笑话', "wavpath": "/mnt/data/hetinggao/vita-e2e/downloads/tmp7m_hpts4.wav"}]
if __name__ == "__main__":
    demo()
