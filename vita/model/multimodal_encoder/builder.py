import yaml
from transformers import WhisperProcessor, WhisperForAudioClassification
from .clip.clip_encoder import CLIPVisionTower
from .eva_clip.eva_clip_encoder import EvaClipVisionTower
from .internvit.internvit_encoder import InternViTVisionTower
from .siglip.siglip_encoder import SiglipVisionTower, SiglipVisionTowerS2
from .whale.init_model import init_model

def build_audio_encoder(audio_encoder_config, **kwargs):
    audio_encoder_name_or_path = audio_encoder_config.mm_audio_encoder
    if "whisper" in audio_encoder_name_or_path.lower():
        whisper = WhisperForAudioClassification.from_pretrained(audio_encoder_name_or_path)
        processor = WhisperProcessor.from_pretrained(
            audio_encoder_config.mm_audio_encoder,
            cache_dir=audio_encoder_config.cache_dir,
        )
        del whisper.projector, whisper.classifier
        audio_encoder = whisper.encoder
        audio_encoder.audio_processor = processor
    elif "audio-encoder_Mixtral-8x7B_New_dim3584" in audio_encoder_name_or_path.lower() or \
        "audio-encoder-qwen2-7b-instruct" in audio_encoder_name_or_path.lower():
        with open(audio_encoder_config.mm_audio_encoder + "/train.yaml", "r") as fin:
            configs = yaml.load(fin, Loader=yaml.FullLoader)

        configs["cmvn_file"] = audio_encoder_config.mm_audio_encoder + "/global_cmvn"

        configs["model_conf"]["freeze_encoder"] = getattr(
            audio_encoder_config, "freeze_audio_encoder", True
        )
        configs["model_conf"]["freeze_adpter"] = getattr(
            audio_encoder_config, "freeze_audio_encoder_adapter", True
        )
        audio_encoder = init_model(configs)
    else:
        raise ValueError(f"Unknown audio encoder: {audio_encoder_name_or_path}")


    return audio_encoder


def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(
        vision_tower_cfg, "mm_vision_tower", getattr(vision_tower_cfg, "vision_tower", None)
    )
    use_s2 = getattr(vision_tower_cfg, "use_s2", False)

    if "sig" in vision_tower.lower():
        if use_s2:
            return SiglipVisionTowerS2(vision_tower, args=vision_tower_cfg, **kwargs)
        else:
            return SiglipVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    elif "eva" in vision_tower.lower():
        if use_s2:
            raise ValueError(f"Currently not supporting S2 for EVA-CLIP")
        else:
            return EvaClipVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    elif "clip" in vision_tower.lower():
        if use_s2:
            raise ValueError(f"Currently not supporting S2 for CLIP")
        else:
            return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    elif "internvit" in vision_tower.lower():
        if use_s2:
            raise ValueError(f"Currently not supporting S2 for InternViT")
        else:
            return InternViTVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    else:
        raise ValueError(f"Unknown vision tower: {vision_tower}")

