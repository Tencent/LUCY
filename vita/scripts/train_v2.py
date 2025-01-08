import os
import pathlib
import torch
import transformers
from typing import Optional, List
from dataclasses import dataclass, field
from transformers import AutoTokenizer, AddedToken
from vita.util import set_random_seed, rank0_print
from vita.util import data_util_v2 as data_util
from vita.model.language_model.vita_qwen2_v2 import VITAQwen2ForCausalLM
from vita.scripts.trainer import VITATrainer, get_mm_adapter_state_maybe_zero_3
from vita.constants import AUDIO_PH, PAD_TOKEN

torch.multiprocessing.set_sharing_strategy('file_system')
local_rank = None

@dataclass
class ModelArguments:
    model_type: Optional[str] = field(default=None)
    model_name_or_path: Optional[str] = field(default="/mnt/data/hetinggao/models/Qwen2-1.5B")
    audio_encoder: Optional[str] = field(default="openai/whisper-medium")
    model_hidden_size: Optional[int] = field(default=1536)
    freeze_backbone: Optional[bool] = field(default=True)
    freeze_audio_encoder: Optional[bool] = field(default=True)
    freeze_tts_adapter: Optional[bool] = field(default=True)
    freeze_audio_encoder_adapter: Optional[bool] = field(default=True)
    freeze_lm_head: Optional[bool] = field(default=False)
    freeze_embed_tokens: Optional[bool] = field(default=True)
    audio_encoder_hidden_size: Optional[int] = field(default=1024)
    text_vocab_size: Optional[int] = field(default=151936)
    text_special_tokens: Optional[int] = field(default=64)
    audio_vocab_size: Optional[int] = field(default=4096)
    audio_special_tokens: Optional[int] = field(default=64)
    audio_projector_hidden_size: Optional[int] = field(default=7168)
    audio_num_codebook: Optional[int] = field(default=7)
    text_additional: Optional[List[str]] = field(default_factory=lambda: ["EOT", "PAD_T", "BOT", "ANS_T", "TTS"])
    audio_additional: Optional[List[str]] = field(default_factory=lambda: ["EOT", "PAD_T", "BOT", "ANS_T", "TTS"])
    cache_dir: Optional[str] = field(default=None)
    model_max_length: int = field(
        default=32768,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    tune_text_embed: Optional[bool] = field(default=False)
    tie_word_embeddings: Optional[bool] = field(default=False)
    loss_reduction: Optional[str] = field(default="sum")
    post_tts_adapter: Optional[bool] = field(default=False)
    initialize_additional_modules: Optional[bool] = field(default=False) # set to true only in stage 1
    post_tts_adapter_num_layers: Optional[int] = field(default=7)

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    # cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    bits: int = field(default=16, metadata={"help": "How many bits to use."})
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."},
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."},
    )
    
    seed: int = field(default=42, metadata={"help": "Random seed."})
    mm_projector_lr: Optional[float] = field(default=None)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    # Only save Adapter
    keys_to_match = ["mm_projector", "embed_tokens", "lm_head"]
    weight_to_save = get_mm_adapter_state_maybe_zero_3(
        trainer.model.named_parameters(), keys_to_match
    )
    trainer.model.config.save_pretrained(output_dir)

    current_folder = output_dir.split("/")[-1]
    parent_folder = os.path.dirname(output_dir)
    if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
        if current_folder.startswith("checkpoint-"):
            mm_projector_folder = os.path.join(parent_folder, "mm_projector")
            os.makedirs(mm_projector_folder, exist_ok=True)
            torch.save(
                weight_to_save, os.path.join(mm_projector_folder, f"{current_folder}.bin")
            )
        else:
            torch.save(weight_to_save, os.path.join(output_dir, f"mm_projector.bin"))
    return


def train():
    global local_rank
    parser = transformers.HfArgumentParser((ModelArguments, data_util.DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    # synchronize common arguments
    data_util.sync_data_args(model_args, data_args)
    print(model_args)
    print(data_args)
    print(training_args)

    local_rank = training_args.local_rank
    if training_args.bf16:
        torch_dtype = torch.bfloat16
    elif training_args.fp16:
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32
    set_random_seed(training_args.seed)
    rank0_print("Start training...")
    
    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig

        bnb_model_from_pretrained_args.update(
            dict(
                device_map={"": training_args.device},
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=training_args.bits == 4,
                    load_in_8bit=training_args.bits == 8,
                    llm_int8_skip_modules=["mm_projector"],
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False,
                    bnb_4bit_compute_dtype=torch_dtype,
                    bnb_4bit_use_double_quant=training_args.double_quant,
                    bnb_4bit_quant_type=training_args.quant_type,  # {'fp4', 'nf4'}
                ),
            )
        )
    
    text_tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        model_max_length=model_args.model_max_length,
        padding_side="right",
        use_fast=True,
    )
    audio_placeholder = AddedToken(AUDIO_PH)
    pad_token = AddedToken(PAD_TOKEN)
    text_tokenizer.add_special_tokens({"additional_special_tokens": [audio_placeholder]})
    text_tokenizer.add_special_tokens({"additional_special_tokens": [pad_token]})


    if model_args.model_type == "qwen2":
        model = VITAQwen2ForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            torch_dtype=torch_dtype,
            attn_implementation="flash_attention_2", # "sdpa", # 
            **bnb_model_from_pretrained_args,
        )
    else:
        raise ValueError(f"Unknown model type {model_args.model_type}")
    model.config.use_cache = False

    if model_args.initialize_additional_modules:
        print("Initializing additional modules")
        model.get_model().initialize_audio_modules(model_args=model_args)
        model.get_model().initialize_extended_embedding(model_args=model_args)
        model.get_model().initialize_tts_adapter(model_args=model_args)
        model.initialize_lm_head(model_args=model_args)

    print("initializing additional modules...")
    model.initialize_additional_configs(model_args)
    model.model.requires_grad_(not model_args.freeze_backbone)
    model.model.audio_encoder.requires_grad_(not model_args.freeze_audio_encoder)
    model.model.embed_tokens.requires_grad_(not model_args.freeze_embed_tokens)
    model.model.audio_mm_projector.requires_grad_(not model_args.freeze_audio_encoder_adapter)
    model.lm_head.requires_grad_(
        (model_args.tie_word_embeddings and not model_args.freeze_embed_tokens) or \
        (not model_args.tie_word_embeddings and not model_args.freeze_lm_head)
    )
    if model_args.post_tts_adapter:
        # must be after setting model.model's gradient because part of tts_adapter is appended to model.model
        model.get_tts_adapter().requires_grad_(not model_args.freeze_tts_adapter)
    print("finish initializing additional modules.")
    
    
    model.config.tokenizer_padding_side = text_tokenizer.padding_side
    model.config.tokenizer_model_max_length = text_tokenizer.model_max_length

    if training_args.bits in [4, 8]:
        for name, module in model.named_modules():
            if "norm" in name:
                module = module.to(torch.float32)
            if "lm_head" in name or "embed_tokens" in name:
                if hasattr(module, "weight"):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    audio_encoder = model.get_audio_encoder()
    print("making data modules...")
    data_module = data_util.make_data_module(
        text_tokenizer=text_tokenizer, 
        audio_processor=audio_encoder.audio_processor, 
        data_args=data_args
    )
    print("finish making data modules and start training...")

    trainer = VITATrainer(model=model, tokenizer=text_tokenizer, args=training_args, **data_module)
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.save_state()

    model.config.use_cache = True
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(training_args.output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(training_args.output_dir, state_dict=cpu_state_dict)  # noqa

if __name__ == "__main__":
    train()
