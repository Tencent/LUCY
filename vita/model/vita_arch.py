from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F

from .multimodal_encoder.builder import build_audio_encoder
from .multimodal_projector.builder import build_audio_projector

class PartiallyFrozenEmbedding(nn.Module):
    def __init__(self, frozen, unfrozen):
        super().__init__()
        self.weight_frozen = frozen
        self.weight_unfrozen = unfrozen
        
    @property
    def weight(self):
        return torch.cat([self.weight_frozen, self.weight_unfrozen])

    def forward(self, idx):
        # weight = torch.cat([self.weight_frozen, self.weight_unfrozen])
        return F.embedding(idx, self.weight)


class VITAMetaModel:
    def __init__(self, config):
        super(VITAMetaModel, self).__init__(config)

        if hasattr(config, "mm_audio_encoder"):
            self.audio_encoder = build_audio_encoder(config)

    def get_audio_encoder(self):
        audio_encoder = getattr(self, "audio_encoder", None)
        return audio_encoder

    def initialize_audio_modules(self, model_args):
        setattr(self.config, "mm_audio_encoder", model_args.audio_encoder)
        setattr(self.config, "mm_audio_num_codebook", model_args.audio_num_codebook)
        setattr(self.config, "mm_audio_encoder_hidden_size", model_args.audio_encoder_hidden_size)
        setattr(self.config, "mm_audio_projector_hidden_size", model_args.audio_projector_hidden_size)
        setattr(self.config, "cache_dir", model_args.cache_dir)
        if self.get_audio_encoder() is None:
            audio_encoder = build_audio_encoder(self.config)
            self.audio_encoder = audio_encoder
            
            audio_projector = build_audio_projector(self.config)
            self.audio_mm_projector = audio_projector
    
    def initialize_extended_embedding(self, model_args):
        # extend embed_tokens with additional audio codec tokens
        std = self.config.initializer_range
        total_vocab_size = (
            model_args.text_vocab_size + model_args.text_special_tokens + \
            (model_args.audio_vocab_size + model_args.audio_special_tokens) * model_args.audio_num_codebook
        )
        setattr(self.config, "total_vocab_size", total_vocab_size)
        setattr(self.config, "text_vocab_size", model_args.text_vocab_size)
        setattr(self.config, "text_special_tokens", model_args.text_special_tokens)
        setattr(self.config, "audio_vocab_size", model_args.audio_vocab_size)
        setattr(self.config, "audio_special_tokens", model_args.audio_special_tokens)
        
        setattr(self.config, "text_vocab_size_padded", model_args.text_vocab_size+model_args.text_special_tokens)
        setattr(self.config, "audio_vocab_size_padded", model_args.audio_vocab_size+model_args.audio_special_tokens)
        
        extended_embed = nn.Parameter(torch.zeros(total_vocab_size - model_args.text_vocab_size, model_args.model_hidden_size))
        extended_embed.data.normal_(mean=0.0, std=std)
        original_embed = nn.Parameter(self.embed_tokens.weight.data.clone()).requires_grad_(False)
        extended_embed_tokens = PartiallyFrozenEmbedding(
            original_embed, extended_embed
        )
        assert extended_embed_tokens.weight.shape[0] == self.config.total_vocab_size
        del self.embed_tokens
        self.embed_tokens = extended_embed_tokens
        


class VITAMetaForCausalLM(ABC):
    @abstractmethod
    def get_model(self):
        pass

    def get_audio_encoder(self):
        return self.get_model().get_audio_encoder()

    def initialized_lm_head(self):
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.total_vocab_size, bias=False)