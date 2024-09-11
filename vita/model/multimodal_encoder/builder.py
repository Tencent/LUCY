from transformers import WhisperProcessor, WhisperForAudioClassification

def build_audio_encoder(audio_encoder_config, **kwargs):
    whisper = WhisperForAudioClassification.from_pretrained(audio_encoder_config.mm_audio_encoder)
    processor = WhisperProcessor.from_pretrained(
        audio_encoder_config.mm_audio_encoder,
        cache_dir=audio_encoder_config.cache_dir,
    )
    del whisper.projector, whisper.classifier
    audio_encoder = whisper.encoder
    audio_encoder.audio_processor = processor
    return audio_encoder
