import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, List, Union, Tuple

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    Qwen2Config,
    Qwen2Model,
    Qwen2ForCausalLM,
)
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast
from transformers.utils import logging
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask, _prepare_4d_causal_attention_mask_for_sdpa
from vita.model.vita_arch import VITAMetaModel, VITAMetaForCausalLM
from vita.constants import IGNORE_INDEX
logger = logging.get_logger(__name__)

class VITAQwen2Config(Qwen2Config):
    model_type = "vita-qwen2"

def qwen2_custom_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    apply_norm: Optional[bool] = True
) -> Union[Tuple, BaseModelOutputWithPast]:
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache

    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # retrieve input_ids and inputs_embeds
    if input_ids is not None and inputs_embeds is not None:
        raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
    elif input_ids is not None:
        batch_size, seq_length = input_ids.shape
    elif inputs_embeds is not None:
        batch_size, seq_length, _ = inputs_embeds.shape
    else:
        raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

    if self.gradient_checkpointing and self.training:
        if use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
            )
            use_cache = False

    past_key_values_length = 0

    if use_cache:
        use_legacy_cache = not isinstance(past_key_values, Cache)
        if use_legacy_cache:
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
        past_key_values_length = past_key_values.get_usable_length(seq_length)

    if position_ids is None:
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        position_ids = torch.arange(
            past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
        )
        position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
    else:
        position_ids = position_ids.view(-1, seq_length).long()

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if attention_mask is not None and self._attn_implementation == "flash_attention_2" and use_cache:
        is_padding_right = attention_mask[:, -1].sum().item() != batch_size
        if is_padding_right:
            raise ValueError(
                "You are attempting to perform batched generation with padding_side='right'"
                " this may lead to unexpected behaviour for Flash Attention version of Qwen2. Make sure to "
                " call `tokenizer.padding_side  = 'left'` before tokenizing the input. "
            )

    if self._attn_implementation == "flash_attention_2":
        # 2d mask is passed through the layers
        attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
    elif self._attn_implementation == "sdpa" and not output_attentions:
        # output_attentions=True can not be supported when using SDPA, and we fall back on
        # the manual implementation that requires a 4D causal mask in all cases.
        attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
            attention_mask,
            (batch_size, seq_length),
            inputs_embeds,
            past_key_values_length,
            sliding_window=self.config.sliding_window,
        )
    else:
        # 4d mask is passed through the layers
        attention_mask = _prepare_4d_causal_attention_mask(
            attention_mask,
            (batch_size, seq_length),
            inputs_embeds,
            past_key_values_length,
            sliding_window=self.config.sliding_window,
        )

    hidden_states = inputs_embeds

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = None

    for decoder_layer in self.layers:
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if self.gradient_checkpointing and self.training:
            layer_outputs = self._gradient_checkpointing_func(
                decoder_layer.__call__,
                hidden_states,
                attention_mask,
                position_ids,
                past_key_values,
                output_attentions,
                use_cache,
            )
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

        hidden_states = layer_outputs[0]

        if use_cache:
            next_decoder_cache = layer_outputs[2 if output_attentions else 1]

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    if apply_norm:
        hidden_states = self.norm(hidden_states)

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    next_cache = None
    if use_cache:
        next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache

    if not return_dict:
        return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )


Qwen2Model.forward = qwen2_custom_forward

class VITAQwen2Model(VITAMetaModel, Qwen2Model):
    config_class = VITAQwen2Config

    def __init__(self, config: Qwen2Config):
        super(VITAQwen2Model, self).__init__(config)


@dataclass
class VITACausalLMOutputWithPast(CausalLMOutputWithPast):
    loss_text: Optional[torch.Tensor] = None
    loss_audios: Optional[torch.Tensor] = None
    loss_states: Optional[torch.Tensor] = None
    tasks: Optional[List[str]] = None


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

    def replace_with_whisper_feature(self, audio_features, inputs_embeds, audio_lengths, audio_attention_mask):
        audio_features_cat = torch.cat([
            audio_feat[:audio_leng] for audio_feat, audio_leng in zip(audio_features, audio_lengths)
        ], dim=0) # Ta x 1024
        audio_num_codebook = self.config.mm_audio_num_codebook
        inputs_embeds[audio_attention_mask] = torch.cat([
            audio_features_cat[:,None,:].expand(-1,audio_num_codebook,-1), # Ta x 7 x H
            inputs_embeds[audio_attention_mask][:,-1:] # Ta x 1 x H 
        ], dim=1) # Ta x 8 x H
        return inputs_embeds

    def forward(
        self,
        input_ids: torch.LongTensor = None, # B x T x L 
        labels: torch.LongTensor = None, # B x T x L 
        attention_mask: Optional[torch.Tensor] = None, # B x T
        audio_attention_mask: Optional[torch.Tensor] = None, # B x T
        audio_feature_lengths: Optional[torch.Tensor] = None,
        audio_lengths: Optional[torch.LongTensor] = None, # B
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        audios: Optional[torch.Tensor] = None,
        states: Optional[torch.Tensor] = None,
        state_attention_mask: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
        tasks: Optional[List[str]] = None,
        indices: Optional[torch.LongTensor] = None,
        dids: Optional[torch.LongTensor] = None,
        idxs: Optional[torch.LongTensor] = None,
        max_input_length: Optional[int] = 1500,
        state_start: Optional[int] = None, 
        state_end: Optional[int] = None,
        infer: Optional[bool] = False,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        post_tts_adapter = getattr(self.config, "post_tts_adapter", False)
        audio_num_codebook = self.config.mm_audio_num_codebook
        inputs_embeds = self.model.embed_tokens(input_ids) # B x T x L x H

        if audios is not None and audios.numel() > 0: # if contains asr task in batch
            if self.config.mm_audio_encoder_type == "whisper":
                audio_features = self.model.audio_encoder(audios).last_hidden_state # B x 80 x 3000 => B x T x 1024
                audio_features = self.model.audio_mm_projector(audio_features)
            elif self.config.mm_audio_encoder_type == "whale":
                audio_input_dict = self.model.audio_encoder(audios, audio_feature_lengths) # B x 80 x 3000 => B x T x 1024
                audio_features = audio_input_dict["inputs_embeds"]
                audio_features = self.model.audio_mm_projector(audio_features)

                assert (audio_attention_mask.sum() == audio_lengths.sum()).all(), \
                    f"audio input length {audio_attention_mask.sum()} vs precomputed audio_length {audio_lengths.sum()}"
            inputs_embeds = self.replace_with_whisper_feature(
                audio_features, inputs_embeds, audio_lengths, audio_attention_mask
            )
            dummy_audio_encoder_loss = 0.
        elif not infer:
            if self.config.mm_audio_encoder_type == "whisper":
                dummy_audio_input = torch.zeros(1, 80, 3000).to(inputs_embeds)
                dummy_audio_features = self.model.audio_encoder(dummy_audio_input).last_hidden_state
                dummy_audio_features = self.model.audio_mm_projector(dummy_audio_features)

                dummy_logits = dummy_audio_features.view(-1, dummy_audio_features.shape[-1]).mean(dim=0) # 1 x H
                dummy_labels = input_ids.new_zeros(1,)

                dummy_audio_encoder_loss = self.compute_loss(dummy_logits, dummy_labels) * 0.

            elif self.config.mm_audio_encoder_type == "whale":
                dummy_audios = torch.zeros(1, 20, 80).to(inputs_embeds)
                dummy_audio_feature_lengths = torch.LongTensor([20]).to(input_ids)
                dummy_audio_features = self.model.audio_encoder(
                    dummy_audios, dummy_audio_feature_lengths
                )["inputs_embeds"]
                dummy_audio_features = self.model.audio_mm_projector(dummy_audio_features)

                dummy_logits = dummy_audio_features.view(-1, dummy_audio_features.shape[-1]).mean(dim=0) # 1 x H
                dummy_labels = input_ids.new_zeros(1,)

                dummy_audio_encoder_loss = self.compute_loss(dummy_logits, dummy_labels) * 0.
        else:
            dummy_audio_encoder_loss = 0.

        inputs_embeds = torch.mean(inputs_embeds, dim=2) # B x T x L x H => B x T x H

        if getattr(self.config, "scale_embeddings", False):
            inputs_embeds = inputs_embeds * (self.config.n_embd**0.5)

        #print(input_ids.shape, inputs_embeds.shape, attention_mask.sum(-1), "indices", indices, "dids", dids, "idxs", idxs)
        
        outputs = self.model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=self.config.use_cache if use_cache is None else use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
            apply_norm=not post_tts_adapter # do not apply norm if use post tts adapter
        )

        text_vocab_size_padded = self.config.text_vocab_size_padded
        audio_vocab_size_padded = self.config.audio_vocab_size_padded

        if not post_tts_adapter:
            hidden_states = outputs.last_hidden_state
            logits = self.lm_head(hidden_states) # B x T x H
        else:
            hidden_states_text = self.model.norm(outputs.hidden_states[self.config.num_hidden_layers])
            hidden_states_audio = self.model.post_tts_module.norm(outputs.hidden_states[-1])
            logits_text = hidden_states_text @ self.lm_head.weight[:text_vocab_size_padded].T
            logits_audio = hidden_states_audio @ self.lm_head.weight[text_vocab_size_padded:].T
            logits = torch.cat([logits_text, logits_audio], dim=-1)

        loss, loss_text, loss_audios, loss_states = None, None, None, None
        if labels is not None:
            logits_text = logits[..., :-1, :text_vocab_size_padded].contiguous()
            labels_text = labels[..., 1:, -1].contiguous()
            loss_text = self.compute_loss(logits_text, labels_text)
            # loss_audios, loss_audios_report = [], []
            loss_audios = []
            for i in range(audio_num_codebook):
                code_start = text_vocab_size_padded+audio_vocab_size_padded * i
                code_end = text_vocab_size_padded+audio_vocab_size_padded * (i + 1)
                logits_audio_i = logits[..., :-1, code_start:code_end].contiguous()
                labels_audio_i = labels[..., 1:, i].contiguous()
                if (labels[...,i] == IGNORE_INDEX).all():
                    continue
                loss_audio_i = self.compute_loss(logits_audio_i, labels_audio_i)
                loss_audios.append(loss_audio_i)

            # import pdb; pdb.set_trace()
            loss_states = []
            if states is not None:
                assert state_start is not None
                assert state_end is not None
                assert state_attention_mask is not None
                logits_state = logits[state_attention_mask][...,state_start:state_end]
                loss_state = self.compute_loss(logits_state, states)
                loss_states.append(loss_state)


            losses = [loss_text] + loss_audios + loss_states
            loss_weights = torch.tensor(
                getattr(self.config, "loss_weights", [1.,1.,1.,1.,1.,1.,1.,1.,1.])
            ).to(loss_text)[:len(losses)]

            losses = [l * w for l, w in zip(losses, loss_weights)]

            if self.config.loss_reduction == "mean":
                loss = sum(losses) / len(losses)
            elif self.config.loss_reduction == "sum":
                loss = sum(losses)
            else:
                raise ValueError(f"{self.config.loss_reduction} not implemented")
            if len(loss_audios) > 0:
                loss_audios = torch.stack(loss_audios)

            if len(loss_states) > 0:
                loss_states = torch.stack(loss_states)

            loss += dummy_audio_encoder_loss

        return VITACausalLMOutputWithPast(
            loss=loss,
            loss_text=loss_text,
            loss_audios=loss_audios,
            loss_states=loss_states,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=outputs.attentions,
            tasks=tasks
        )

    def codec_layer_shift_reverse(self, shifted_input_id, layer):
        input_id = shifted_input_id - self.config.text_vocab_size_padded - layer * self.config.audio_vocab_size_padded
        return input_id

    def compute_loss(self, logits, labels):
        *_, vocab_size = logits.shape
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, vocab_size), labels.view(-1))
        return loss

    def forward_text(
        self,
        input_ids: torch.LongTensor = None, # B x T
        attention_mask: Optional[torch.Tensor] = None, # B x T
        audio_lengths: Optional[torch.LongTensor] = None, # B
        position_ids: Optional[torch.LongTensor] = None,
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
    ):
        PAD_A = self.config.additional_tokens["PAD_A"]
        ANS_A = self.config.additional_tokens["ANS_A"]
        BOT = self.config.additional_tokens["BOT"]
        EOT = self.config.additional_tokens["EOT"]
        task = "TQA"
        TASK_TOK = self.config.additional_tokens[task]
        audio_num_codebook = self.config.mm_audio_num_codebook
        text_vocab_size_padded = self.config.text_vocab_size_padded
        # assume batch size is 1
        device = input_ids.device
        if past_key_values is None:
            text = input_ids[0]
            text_leng = len(text)

            codec_by_layer_padded = torch.zeros(
                audio_num_codebook, text_leng+3, dtype=torch.long
            ).to(device) # 1 for BOT and 1 for EOT and 1 for ANS
            for i in range(audio_num_codebook):
                codec_by_layer_padded[i] = torch.LongTensor(
                    [
                        self.codec_layer_shift(PAD_A, i)
                    ] * (len(text)+2) + [ # 1 for BOT and 1 for EOT
                        self.codec_layer_shift(ANS_A, i)
                    ]
                )
            text_padded = torch.cat([
                torch.LongTensor([BOT]).to(device), 
                text, 
                torch.LongTensor([EOT, TASK_TOK]).to(device)
            ]).unsqueeze(0) # 1 x (T+3)
            input_length = len(text) + 3
            input_ids = torch.cat([codec_by_layer_padded, text_padded], dim=0) # 8 x input_length
            attention_mask = torch.ones((1,input_length), dtype=bool).to(device)
        else:
            text = input_ids[0]
            text_leng = len(text)
            codec_by_layer_padded = torch.zeros(audio_num_codebook, text_leng, dtype=torch.long).to(device)
            for i in range(audio_num_codebook):
                codec_by_layer_padded[i] = self.codec_layer_shift(PAD_A, i)
            input_ids = torch.cat([codec_by_layer_padded, text.unsqueeze(0)], dim=0)
            input_length = text_leng
            attention_mask = torch.ones((1,input_length), dtype=bool).to(device)

        input_ids = input_ids.T.unsqueeze(0)

        outputs = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            audios=None,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
        )

        logits = outputs.logits[...,:text_vocab_size_padded]
        return VITACausalLMOutputWithPast(
            loss=None,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def codec_layer_shift(self, input_id, layer):
        text_vocab_size_padded = self.config.text_vocab_size_padded
        audio_vocab_size_padded = self.config.audio_vocab_size_padded
        return input_id + text_vocab_size_padded + layer * audio_vocab_size_padded
        

AutoConfig.register("vita-qwen2", VITAQwen2Config)
AutoModelForCausalLM.register(VITAQwen2Config, VITAQwen2ForCausalLM)
