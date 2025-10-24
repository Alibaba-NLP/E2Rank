from dataclasses import dataclass, field
from typing import Optional, Sequence, Dict, Union
from transformers import TrainingArguments as HFTrainingArguments


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )

    # model parameters
    use_embed_loss: bool = field(
        default=True, metadata={"help": "Use embedding loss for training"}
    )
    loss_embed_factor: float = field(
        default=1.0, metadata={"help": "Factor to multiply the generative loss by"}
    )
    temperature: float = field(
        default=0.03, metadata={"help": "Temperature for scaling the logits"}
    )
    use_ranknet_loss: bool = field(
        default=False, metadata={"help": "Use ranknet loss for training"}
    )
    loss_ranknet_factor: float = field(
        default=1.0, metadata={"help": "Factor to multiply the ranknet loss by"}
    )
    ranknet_scale_factor: float = field(
        default=5.0, metadata={"help": "Factor to multiply the ranknet score by"}
    )

    negatives_cross_device: Optional[bool] = field(default=False)


@dataclass
class DataArguments:
    data_path: str = field(
        metadata={"help": "Path to the dataset"}
    )
    data_type: str = field(
        default="sft", metadata={"help": "Type of the dataset"}
    )
    use_full_doc: bool = field(
        default=False, metadata={"help": "Whether to use full document for training"}
    )
    use_listwise: bool = field(
        default=True, metadata={"help": "Whether to use listwise prompt for training"}
    )
    data_cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the preprocessed data"}
    )
    d_max_len: int = field(
        default=512, metadata={"help": "Max length of the passage"}
    )
    q_max_len: int = field(
        default=32, metadata={"help": "Max length of the query"}
    )
    num_negatives: int = field(
        default=1, metadata={"help": "Number of negatives per query"}
    )


@dataclass
class TrainingArguments(HFTrainingArguments):
    pass


@dataclass
class LoraArguments:
    lora_enabled: bool = False
    lora_path: Optional[str] = None
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"]
    )
    lora_bias: str = "none"
