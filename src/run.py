import logging
import os
import sys
import pathlib
import torch
from transformers import AutoConfig, AutoTokenizer, AutoModel, AutoModelForCausalLM
from transformers import HfArgumentParser, set_seed, Trainer as HFTrainer
from peft import LoraConfig, get_peft_model, PeftModel

from arguments import ModelArguments, DataArguments, TrainingArguments, LoraArguments
from data import MixedRankDataset, MixedRankDataCollator
from modeling import JointTrainingModel
from utils import *


logger = logging.getLogger(__name__)


def safe_save_model_for_hf_trainer(trainer: HFTrainer, output_dir: str):
    """Collects the state dict and dump to disk."""

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def main():
    parser = HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoraArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args, lora_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, lora_args = parser.parse_args_into_dataclasses()

    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("MODEL parameters %s", model_args)

    set_seed(training_args.seed)

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        trust_remote_code=True,
        cache_dir=model_args.cache_dir,
    )

    model = AutoModel.from_pretrained(
        model_args.model_name_or_path,
        config=config, cache_dir=model_args.cache_dir,
        trust_remote_code=True,
        attn_implementation="flash_attention_2"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        padding_side="left",
        cache_dir=model_args.cache_dir,
        trust_remote_code=True,
    )

    # Load Lora
    if lora_args.lora_enabled:
        if lora_args.lora_path:
            print(f"Loading Lora from {lora_args.lora_path}")
            model = PeftModel.from_pretrained(
                model,
                lora_args.lora_path,
                is_trainable=True,
            )
        else:
            print("Initializing LoRA")
            lora_config = LoraConfig(
                r=lora_args.lora_r,
                lora_alpha=lora_args.lora_alpha,
                target_modules=lora_args.lora_target_modules,
                lora_dropout=lora_args.lora_dropout,
                bias=lora_args.lora_bias,
                task_type="FEATURE_EXTRACTION",
            )
            model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    model = JointTrainingModel(
        model=model,
        tokenizer=tokenizer,
        temperature=model_args.temperature,
        use_embed_loss=model_args.use_embed_loss,
        use_ranknet_loss=model_args.use_ranknet_loss,
        loss_embed_factor=model_args.loss_embed_factor,
        loss_ranknet_factor=model_args.loss_ranknet_factor,
        ranknet_scale_factor=model_args.ranknet_scale_factor,
        negatives_cross_device=model_args.negatives_cross_device,
    )

    model.train()

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

        training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}

    class Trainer(HFTrainer):

        def _save(self, output_dir=None, state_dict=None):
            output_dir = output_dir if output_dir is not None else self.args.output_dir
            os.makedirs(output_dir, exist_ok=True)
            print(f"Saving model checkpoint to {output_dir}")

            if self.is_deepspeed_enabled:
                model_to_save = self.deepspeed.model
            else:
                model_to_save = self.model.model
            model_to_save.save_pretrained(
                output_dir, safe_serialization=self.args.save_safetensors, 
                state_dict={key.removeprefix("model."): value for key, value in state_dict.items() if key.startswith("model.")}
            )
    
            if self.tokenizer is not None and self.is_world_process_zero():
                self.tokenizer.save_pretrained(output_dir, safe_serialization=self.args.save_safetensors)

    train_dataset = MixedRankDataset(data_args=data_args)
    data_collator = MixedRankDataCollator(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")) and not training_args.overwrite_output_dir:
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    safe_save_model_for_hf_trainer(
        trainer=trainer, output_dir=training_args.output_dir)
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(training_args.output_dir)
        torch.save(model_args, os.path.join(training_args.output_dir, "model_args.bin"))
        torch.save(training_args, os.path.join(training_args.output_dir, "training_args.bin"))
    print("Training done.")


if __name__ == "__main__":
    main()
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()
    print("Success.")
