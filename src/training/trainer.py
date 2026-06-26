import logging
import os

import torch
from datasets import load_dataset
from omegaconf import DictConfig
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.trainer_utils import get_last_checkpoint
from trl import DPOConfig, DPOTrainer, SFTConfig, SFTTrainer

logger = logging.getLogger(__name__)


def load_model_and_tokenizer(model_cfg: DictConfig, is_train: bool = True):
    """Load tokenizer and model with given configuration."""
    logger.info(f"Loading model {model_cfg.name}...")

    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg.name, trust_remote_code=model_cfg.trust_remote_code
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    kwargs = {
        "torch_dtype": getattr(torch, model_cfg.torch_dtype),
        "trust_remote_code": model_cfg.trust_remote_code,
    }

    load_in_8bit = model_cfg.get("load_in_8bit", False)
    load_in_4bit = model_cfg.get("load_in_4bit", False)

    if torch.cuda.is_available():
        if model_cfg.device_map == "auto" and (load_in_8bit or load_in_4bit):
            from accelerate import Accelerator

            kwargs["device_map"] = {"": Accelerator().local_process_index}
        else:
            kwargs["device_map"] = model_cfg.device_map

        if load_in_8bit or load_in_4bit:
            compute_dtype = getattr(
                torch, model_cfg.get("bnb_4bit_compute_dtype", "bfloat16")
            )

            quantization_config = BitsAndBytesConfig(
                load_in_8bit=load_in_8bit,
                load_in_4bit=load_in_4bit,
                # QLoRA settings
                bnb_4bit_quant_type=model_cfg.get("bnb_4bit_quant_type", "nf4"),
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=model_cfg.get(
                    "bnb_4bit_use_double_quant", True
                ),
            )

            kwargs["quantization_config"] = quantization_config
    else:
        # Prevent meta device offloading on Mac/CPU which crashes PEFT backward pass
        if torch.backends.mps.is_available():
            kwargs["device_map"] = {"": "mps"}
        else:
            kwargs["device_map"] = {"": "cpu"}

        if load_in_8bit or load_in_4bit:
            logger.warning(
                "CUDA is not available. Disabling 8-bit/4-bit quantization as "
                "bitsandbytes requires CUDA."
            )

    model = AutoModelForCausalLM.from_pretrained(model_cfg.name, **kwargs)

    if is_train:
        model.config.use_cache = False
        model.gradient_checkpointing_enable()
        # This forces the model to track gradients for the initial inputs so
        # the gradients successfully flow backward to your trainable LoRA adapters
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

        # Apply LoRA
        if "lora" in model_cfg:
            from peft import PeftModel

            if isinstance(model, PeftModel):
                logger.info(
                    "PEFT model detected. Continuing training on existing adapter."
                )
                # Ensure the existing adapter requires gradients
                for name, param in model.named_parameters():
                    if "lora_" in name:
                        param.requires_grad = True
            else:
                lora_cfg = model_cfg.lora
                config = LoraConfig(
                    r=lora_cfg.r,
                    lora_alpha=lora_cfg.lora_alpha,
                    lora_dropout=lora_cfg.lora_dropout,
                    bias=lora_cfg.bias,
                    task_type=lora_cfg.task_type,
                    target_modules=list(lora_cfg.target_modules),
                )
                model = get_peft_model(model, config)
                logger.info("Applied LoRA configuration.")

    return model, tokenizer


def run_sft_training(cfg: DictConfig):
    """Run Supervised Fine-Tuning."""
    logger.info("Initializing SFT Training...")

    model, tokenizer = load_model_and_tokenizer(cfg.model, is_train=True)

    logger.info(
        f"Loading SFT training dataset from {cfg.data.output_sft_train_file}..."
    )
    dataset_train = load_dataset("json", data_files=cfg.data.output_sft_train_file)[
        "train"
    ]

    logger.info(
        f"Loading SFT validation dataset from {cfg.data.output_sft_val_file}..."
    )
    dataset_val = load_dataset("json", data_files=cfg.data.output_sft_val_file)["train"]

    # Format text for training
    def format_chat(example):
        messages = [
            {"role": "system", "content": example["instruction"]},
            {"role": "user", "content": example["input"]},
            {
                "role": "assistant",
                "content": (
                    f"Action: {example['output']['action']}\\n"
                    f"Reasoning: {example['output']['reasoning']}\\n"
                    f"Facets: {example['output']['facets']}\\n"
                    f"Response: {example['output']['response']}"
                ),
            },
        ]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        return {"text": formatted}

    formatted_train = dataset_train.map(
        format_chat, remove_columns=dataset_train.column_names
    )
    formatted_val = dataset_val.map(
        format_chat, remove_columns=dataset_val.column_names
    )

    training_args = SFTConfig(
        output_dir=cfg.training.output_dir,
        per_device_train_batch_size=cfg.training.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        num_train_epochs=cfg.training.num_train_epochs,
        learning_rate=cfg.training.learning_rate,
        warmup_ratio=cfg.training.warmup_ratio,
        bf16=cfg.training.bf16,
        eval_strategy="steps",
        eval_steps=cfg.training.logging_steps,
        logging_steps=cfg.training.logging_steps,
        save_steps=cfg.training.save_steps,
        save_total_limit=cfg.training.save_total_limit,
        optim=cfg.training.optim,
        report_to=cfg.training.report_to,
        run_name="sft_training",
        dataset_text_field="text",
        max_seq_length=cfg.training.max_seq_length,
        packing=cfg.training.packing,
        remove_unused_columns=cfg.training.get("remove_unused_columns", True),
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=formatted_train,
        eval_dataset=formatted_val,
        args=training_args,
    )

    logger.info("Starting SFT Training...")
    last_checkpoint = None
    if cfg.training.get("resume_from_checkpoint", False) and os.path.isdir(cfg.training.output_dir):
        last_checkpoint = get_last_checkpoint(cfg.training.output_dir)
        if last_checkpoint is not None:
            logger.info(f"Resuming SFT training from {last_checkpoint}")

    trainer.train(resume_from_checkpoint=last_checkpoint)

    trainer.save_model(f"{cfg.training.output_dir}/final")
    tokenizer.save_pretrained(f"{cfg.training.output_dir}/final")
    logger.info("SFT Training complete and model saved.")


def run_dpo_training(cfg: DictConfig):
    """Run Direct Preference Optimization."""
    logger.info("Initializing DPO Training...")

    # Load model and reference model
    model, tokenizer = load_model_and_tokenizer(cfg.model, is_train=True)
    ref_model, _ = load_model_and_tokenizer(
        cfg.model, is_train=False
    )  # Ref model without LoRA adapters trainable

    logger.info(
        f"Loading DPO training dataset from {cfg.data.output_dpo_train_file}..."
    )
    dataset_train = load_dataset("json", data_files=cfg.data.output_dpo_train_file)[
        "train"
    ]

    logger.info(
        f"Loading DPO validation dataset from {cfg.data.output_dpo_val_file}..."
    )
    dataset_val = load_dataset("json", data_files=cfg.data.output_dpo_val_file)["train"]

    training_args = DPOConfig(
        output_dir=cfg.training.output_dir,
        per_device_train_batch_size=cfg.training.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        num_train_epochs=cfg.training.num_train_epochs,
        learning_rate=cfg.training.learning_rate,
        warmup_ratio=cfg.training.warmup_ratio,
        bf16=cfg.training.bf16,
        eval_strategy="steps",
        eval_steps=cfg.training.logging_steps,
        logging_steps=cfg.training.logging_steps,
        save_steps=cfg.training.save_steps,
        save_total_limit=cfg.training.save_total_limit,
        optim=cfg.training.optim,
        report_to=cfg.training.report_to,
        run_name="dpo_training",
        beta=cfg.training.beta,
        max_prompt_length=cfg.training.max_prompt_length,
        max_length=cfg.training.max_length,
        remove_unused_columns=cfg.training.get("remove_unused_columns", False),
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=dataset_train,
        eval_dataset=dataset_val,
        tokenizer=tokenizer,
    )

    logger.info("Starting DPO Training...")
    last_checkpoint = None
    if cfg.training.get("resume_from_checkpoint", False) and os.path.isdir(cfg.training.output_dir):
        last_checkpoint = get_last_checkpoint(cfg.training.output_dir)
        if last_checkpoint is not None:
            logger.info(f"Resuming DPO training from {last_checkpoint}")

    trainer.train(resume_from_checkpoint=last_checkpoint)

    trainer.save_model(f"{cfg.training.output_dir}/final")
    tokenizer.save_pretrained(f"{cfg.training.output_dir}/final")
    logger.info("DPO Training complete and model saved.")
