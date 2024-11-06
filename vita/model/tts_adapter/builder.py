import torch.nn as nn
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer, Qwen2RMSNorm

def build_tts_adapter(config, **kwargs):
    tts_adapter = nn.ModuleList([
        Qwen2DecoderLayer(config, config.num_hidden_layers+layer_idx) \
            for layer_idx in range(config.post_tts_adapter_num_layers)
    ])
    tts_norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    post_tts_module = nn.ModuleDict({
        "adapter": tts_adapter,
        "norm": tts_norm,
    })
    return post_tts_module
