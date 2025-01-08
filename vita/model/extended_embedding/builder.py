import torch
import torch.nn as nn
import torch.nn.functional as F
class PartiallyFrozenEmbedding(nn.Module):
    def __init__(self, frozen, unfrozen):
        super().__init__()
        self.weight_frozen = frozen
        self.weight_unfrozen = unfrozen
        
    @property
    def weights(self):
        return torch.cat([self.weight_frozen, self.weight_unfrozen])

    def forward(self, idx):
        return F.embedding(idx, self.weights)

def build_extended_embedding(config, **kwargs):
    std = config.initializer_range
    total_vocab_size = config.total_vocab_size
    text_vocab_size = config.text_vocab_size
    original_weight = kwargs.get("original_weight", torch.zeros(text_vocab_size, config.hidden_size))
    #assert original_weight.shape[0] == text_vocab_size, \
    #    f"original embedding has shape of {original_weight.shape} while text vocab size is {text_vocab_size}"
    #assert original_weight.shape[1] == config.hidden_size, \
    #    f"original embedding has shape of {original_weight.shape} while hidden dimension is {config.hidden_size}"
    if config.tune_text_embed:
        extended_embed = torch.zeros(total_vocab_size - text_vocab_size, config.hidden_size)
        extended_embed.data.normal_(mean=0.0, std=std)
        original_embed = original_weight.clone()
        embed_weight = torch.cat([original_embed, extended_embed])
        extended_embed_tokens = nn.Embedding.from_pretrained(embed_weight, freeze=False)
        print(extended_embed_tokens.weight.shape, total_vocab_size)
    #    assert extended_embed_tokens.weight.shape[0] == total_vocab_size, \
    #        f"extended embedding has shape of {extended_embed_tokens.weight.shape} while total vocab size is {total_vocab_size}"
    else:
        extended_embed = nn.Parameter(torch.zeros(total_vocab_size - text_vocab_size, config.hidden_size))
        extended_embed.data.normal_(mean=0.0, std=std)
        original_embed = nn.Parameter(original_weight.clone()).requires_grad_(False)
        extended_embed_tokens = PartiallyFrozenEmbedding(
            original_embed, extended_embed
        )
        assert extended_embed_tokens.weights.shape[0] == total_vocab_size
    return extended_embed_tokens
