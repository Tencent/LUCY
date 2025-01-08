from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F

from .multimodal_encoder.builder import build_audio_encoder, build_vision_tower
from .multimodal_projector.builder import build_audio_projector, build_vision_projector
from .extended_embedding.builder import build_extended_embedding
from .tts_adapter.builder import build_tts_adapter

from transformers import logging
logger = logging.get_logger(__name__)

def get_audio_encoder_type(audio_encoder):
    if "whisper" in audio_encoder.lower():
        audio_encoder_type = "whisper" 
    elif "audio-encoder-qwen2-7b-instruct" in audio_encoder.lower():
        audio_encoder_type = "whale"
    else:
        raise ValueError(f"Unknown encoder type {model_args.audio_encoder}")
    return audio_encoder_type

class VITAMetaModel:
    def __init__(self, config):
        super(VITAMetaModel, self).__init__(config)

        if hasattr(config, "mm_audio_encoder"):
            self.audio_encoder = build_audio_encoder(config)
            self.audio_mm_projector = build_audio_projector(self.config)
            if getattr(self.config, "mm_audio_encoder_type", None) is None:
                audio_encoder_type = get_audio_encoder_type(config.mm_audio_encoder)
                setattr(self.config, "mm_audio_encoder_type", audio_encoder_type)
        
        if hasattr(config, "total_vocab_size"):
            self.embed_tokens = build_extended_embedding(self.config)

        if getattr(config, "post_tts_adapter", False):
            self.post_tts_module = build_tts_adapter(self.config)
            self.text_norm = self.norm
            self.layers.extend(self.post_tts_module.adapter)

        if False and hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(
                config, delay_load=False
            )
            self.mm_projector = build_vision_projector(config)

    def get_audio_encoder(self):
        audio_encoder = getattr(self, "audio_encoder", None)
        return audio_encoder

    def initialize_audio_modules(self, model_args):
        setattr(self.config, "mm_audio_encoder", model_args.audio_encoder)
        setattr(self.config, "mm_audio_num_codebook", model_args.audio_num_codebook)
        setattr(self.config, "mm_audio_encoder_hidden_size", model_args.audio_encoder_hidden_size)
        setattr(self.config, "mm_audio_projector_hidden_size", model_args.audio_projector_hidden_size)
        audio_encoder_type = get_audio_encoder_type(model_args.audio_encoder)
        setattr(self.config, "mm_audio_encoder_type", audio_encoder_type)
        setattr(self.config, "cache_dir", model_args.cache_dir)
        setattr(self.config, "mm_audio_projector_type", getattr(model_args, "audio_projector_type", "linear"))
        if self.get_audio_encoder() is None:
            audio_encoder = build_audio_encoder(self.config)
            self.audio_encoder = audio_encoder
            
            audio_projector = build_audio_projector(self.config)
            self.audio_mm_projector = audio_projector
        if "audio-encoder-qwen2-7b-instruct" in model_args.audio_encoder.lower():
            print(f"loading weights of {model_args.audio_encoder}")
            checkpoint = torch.load(model_args.audio_encoder + "/final.pt", map_location="cpu")
            model_dict = self.audio_encoder.state_dict()
            for key in model_dict.keys():
                if key in checkpoint.keys():
                    if model_dict[key].shape == checkpoint[key].shape:
                        model_dict[key] = checkpoint[key]
                    else:
                        print(
                            "Key {} has different shape, {} VS {}".format(
                                key, model_dict[key].shape, checkpoint[key].shape
                            )
                        )
                else:
                    print("Key {} has not in resume model".format(key))
            self.audio_encoder.load_state_dict(model_dict)

    
    def initialize_extended_embedding(self, model_args):
        # extend embed_tokens with additional audio codec tokens
        std = self.config.initializer_range
        phone_vocab_size = getattr(model_args, "phone_vocab_size", 0)
        phone_special_tokens = getattr(model_args, "phone_special_tokens", 0)
        
        total_vocab_size = (
            model_args.text_vocab_size + model_args.text_special_tokens + \
            (model_args.audio_vocab_size + model_args.audio_special_tokens) * model_args.audio_num_codebook + \
            phone_vocab_size + phone_special_tokens
        )
        setattr(self.config, "total_vocab_size", total_vocab_size)
        setattr(self.config, "text_vocab_size", model_args.text_vocab_size)
        setattr(self.config, "text_special_tokens", model_args.text_special_tokens)
        setattr(self.config, "audio_vocab_size", model_args.audio_vocab_size)
        setattr(self.config, "audio_special_tokens", model_args.audio_special_tokens)        
        setattr(self.config, "text_vocab_size_padded", model_args.text_vocab_size+model_args.text_special_tokens)
        setattr(self.config, "audio_vocab_size_padded", model_args.audio_vocab_size+model_args.audio_special_tokens)
        setattr(self.config, "vocab_size", total_vocab_size)
        setattr(self.config, "tune_text_embed", model_args.tune_text_embed)
        if phone_vocab_size > 0 and phone_special_tokens > 0:
            setattr(self.config, "phone_vocab_size", phone_vocab_size)
            setattr(self.config, "phone_special_tokens", phone_special_tokens)
            setattr(self.config, "phone_vocab_size_padded", phone_vocab_size+phone_special_tokens)
        extended_embed_tokens = build_extended_embedding(self.config, original_weight=self.embed_tokens.weight.data)
        del self.embed_tokens
        self.embed_tokens = extended_embed_tokens

    def initialize_tts_adapter(self, model_args):
        setattr(self.config, "post_tts_adapter", model_args.post_tts_adapter)
        if self.config.post_tts_adapter:
            setattr(self.config, "post_tts_adapter_num_layers", model_args.post_tts_adapter_num_layers)
            self.post_tts_module = build_tts_adapter(self.config)
            self.text_norm = self.norm
            self.layers.extend(self.post_tts_module.adapter)
            #from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer, Qwen2RMSNorm
            #tts_adapter = nn.ModuleList([
            #    Qwen2DecoderLayer(self.config, self.config.num_hidden_layers+layer_idx) \
            #        for layer_idx in range(self.config.post_tts_adapter_num_layers)
            #])
            #self.layers.extend(tts_adapter)
            #self.text_norm = self.norm
            #tts_norm = Qwen2RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)
            #self.post_tts_module = nn.ModuleDict({
            #    "adapter": tts_adapter,
            #    "norm": tts_norm,
            #})


class VITAMetaForCausalLM(ABC):
    @abstractmethod
    def get_model(self):
        pass

    def get_audio_encoder(self):
        return self.get_model().get_audio_encoder()

    def get_tts_adapter(self):
        return getattr(self.get_model(), "post_tts_module", None)

    def initialize_lm_head(self, model_args):
        setattr(self.config, "tie_word_embeddings", model_args.tie_word_embeddings)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.total_vocab_size, bias=False)
        if self.config.tie_word_embeddings:
            logger.warning("Tie word embeddings and lm head together.") 
            self.lm_head.weight = self.get_model().embed_tokens.weight


    def initialize_additional_configs(self, model_args):
        loss_reduction = getattr(model_args, "loss_reduction", "sum")
        loss_weights = getattr(model_args, "loss_weights", [1., 1., 1., 1., 1., 1., 1., 1.])
        setattr(self.config, "loss_reduction", loss_reduction)
        text_additional_tokens = {token: model_args.text_vocab_size + i for i, token in enumerate(model_args.text_additional)}
        audio_additional_tokens = {token: model_args.audio_vocab_size + i for i, token in enumerate(model_args.audio_additional)}
        setattr(self.config, "text_additional_tokens", text_additional_tokens)
        setattr(self.config, "audio_additional_tokens", audio_additional_tokens)
        setattr(self.config, "additional_tokens", {**audio_additional_tokens, **text_additional_tokens})
        setattr(self.config, "loss_weights", loss_weights)
