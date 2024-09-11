import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Union, Tuple
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    MixtralConfig,
    MixtralForCausalLM,
    MixtralModel,
    Qwen2Config,
    Qwen2Model,
    Qwen2ForCausalLM,
)
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast, MoeCausalLMOutputWithPast
from transformers.utils import logging, is_torchdynamo_compiling
from vita.model.vita_arch import VITAMetaModel, VITAMetaForCausalLM


class VITAQwen2Config(Qwen2Config):
    model_type = "vita-qwen2"


class VITAQwen2Model(VITAMetaModel, Qwen2Model):
    config_class = VITAQwen2Config

    def __init__(self, config: MixtralConfig):
        super(VITAQwen2Model, self).__init__(config)


class VITAQwen2ForCausalLM(Qwen2ForCausalLM, VITAMetaForCausalLM):
    config_class = VITAQwen2Config
    def __init__(self, config):
        super(Qwen2ForCausalLM, self).__init__(config)
        self.model = VITAQwen2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    # def concat_whisper_feat(self, audio_feature, input_ids, T, task):
    #     batch_size = len(T)
    #     for i in range(batch_size):
    #         if task[i] not in ["TTS"]:
    #             for jth_code in range(7): # first 7 arrays of inputs are audio codes
    #                 input_ids[jth_code][i, 1:T[i]+1, :] = audio_feature[i][: T[i]].clone()
    #     return input_ids

    def replace_with_whisper_feature(self, audio_features, input_embeds, audio_lengths, use_audio_indices):
        # B, T, num_codebook, H = input_embeds.shape
        # import pdb; pdb.set_trace()
        for idx, audio, audio_leng in zip(use_audio_indices, audio_features, audio_lengths):
            # for jth_code in range(1, num_codebook):
            input_embeds[idx,1:audio_leng+1,:-1] = audio[:audio_leng,None,:] # shift 1 for BOA
        return input_embeds

    def forward(
        self,
        input_ids: torch.LongTensor = None, # B x T x L 
        attention_mask: Optional[torch.Tensor] = None, # B x T
        audio_lengths: Optional[torch.LongTensor] = None, # B x T x 80
        position_ids: Optional[torch.LongTensor] = None, # B x T x 8
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        audios: Optional[torch.Tensor] = None,
        use_audio_indices: Optional[List[torch.LongTensor]] = None,
        output_labels_attention_mask: Optional[torch.BoolTensor] = None,
        output_logits_attention_mask: Optional[torch.BoolTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        # num_codebook = len(input_ids)
        
        batch_size, T, num_codebook = input_ids.shape
        audio_num_codebook = num_codebook - 1

        if self.config.tokenizer_model_max_length < T:
            raise ValueError(
                f"Cannot forward sequence of length {T}, max seq length is only {self.config.tokenizer_model_max_length}."
            )

        inputs_embeds = self.model.embed_tokens(input_ids) # B x T x L x H

        audio_features = self.model.audio_encoder(audios).last_hidden_state # B x 80 x 3000 => B x T x 1024
        audio_features = self.model.audio_mm_projector(audio_features)
        inputs_embeds = self.replace_with_whisper_feature(
            audio_features, inputs_embeds, audio_lengths, use_audio_indices
        )

        inputs_embeds = torch.sum(inputs_embeds, dim=2) / len(inputs_embeds) # B x T x L x H => B x T x H

        if getattr(self.config, "scale_embeddings", False):
            x = x * (self.config.n_embd**0.5)

        outputs = self.model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        text_vocab_size_padded = self.config.text_vocab_size_padded
        audio_vocab_size_padded = self.config.audio_vocab_size_padded

        hidden_states = outputs[0]

        logits = self.lm_head(hidden_states)
        
        logits_mask_text = output_logits_attention_mask[...,-1]
        labels_mask_text = output_labels_attention_mask[...,-1]
        logits_text = logits[logits_mask_text][...,:text_vocab_size_padded]
        labels_text = input_ids[...,-1][labels_mask_text]

        loss_audios = []
        loss_text = self.compute_loss(logits_text, labels_text)
        for i in range(audio_num_codebook):
            code_start = text_vocab_size_padded+audio_vocab_size_padded * i
            code_end = text_vocab_size_padded+audio_vocab_size_padded * (i + 1)
            logits_mask_audio = output_logits_attention_mask[...,i]
            labels_mask_audio = output_labels_attention_mask[...,i]
            logits_audio_i = logits[logits_mask_audio][...,code_start:code_end]

            labels_audio_i = self.codec_layer_shift_reserve(input_ids[...,i][labels_mask_audio], i)

            loss_audio_i = self.compute_loss(logits_audio_i, labels_audio_i)
            loss_audios.append(loss_audio_i)

        loss = loss_text + sum(loss_audios)
        return CausalLMOutputWithPast(
            loss=loss,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def codec_layer_shift_reserve(self, shifted_input_id, layer):
        input_id = shifted_input_id - self.config.text_vocab_size_padded - layer * self.config.audio_vocab_size_padded
        return input_id

    def compute_loss(self, logits, labels):
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits, labels)
        return loss

AutoConfig.register("vita-qwen2", VITAQwen2Config)
AutoModelForCausalLM.register(VITAQwen2Config, VITAQwen2ForCausalLM)
