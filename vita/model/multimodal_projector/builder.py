import torch
import torch.nn as nn

# https://github.com/gpt-omni/mini-omni/blob/main/litgpt/model.py
class LinearProjector(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.fc_1 = nn.Linear(config.mm_audio_encoder_hidden_size, config.mm_audio_projector_hidden_size, bias=False)
        self.fc_2 = nn.Linear(config.mm_audio_encoder_hidden_size, config.mm_audio_projector_hidden_size, bias=False)
        self.proj = nn.Linear(config.mm_audio_projector_hidden_size, config.hidden_size, bias=False)

        self.config = config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fc_1 = self.fc_1(x)
        x_fc_2 = self.fc_2(x)
        x = torch.nn.functional.silu(x_fc_1) * x_fc_2
        return self.proj(x)


def build_audio_projector(config, **kwargs):
    projector_type = getattr(config, "audio_projector_type", "linear")

    if projector_type == "linear":
        return LinearProjector(config)

    raise ValueError(f"Unknown projector type: {projector_type}")
