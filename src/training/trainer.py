import torch
import logging
from typing import Optional, Dict, Any
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, DPOTrainer
from omegaconf import DictConfig

logger = logging.getLogger(__name__)

def load_model_and_tokenizer(model_cfg: DictConfig, is_train: bool = True):
    """Load tokenizer and model with given configuration."""
    logger.info(f"Loading model {model_cfg.name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg.name,
        trust_remote_code=model_cfg.trust_remote_code
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg.name,
        torch_dtype=getattr(torch, model_cfg.torch_dtype),
        device_map=model_cfg.device_map,
        trust_remote_code=model_cfg.trust_remote_code,
        load_in_8bit=model_cfg.get("load_in_8bit", False),
        load_in_4bit=model_cfg.get("load_in_4bit", False),
    )
    
    if is_train:
        model.config.use_cache = False
        model.gradient_checkpointing_enable()
        
        # Apply LoRA
        if "lora" in model_cfg:
            lora_cfg = model_cfg.lora
            config = LoraConfig(
                r=lora_cfg.r,
                lora_alpha=lora_cfg.lora_alpha,
                lora_dropout=lora_cfg.lora_dropout,
                bias=lora_cfg.bias,
                task_type=lora_cfg.task_type,
                target_modules=list(lora_cfg.target_modules)
            )
            model = get_peft_model(model, config)
            logger.info("Applied LoRA configuration.")
            
    return model, tokenizer

def run_sft_training(cfg: DictConfig):
    """Run Supervised Fine-Tuning."""
    logger.info("Initializing SFT Training...")
    
    model, tokenizer = load_model_and_tokenizer(cfg.model, is_train=True)
    
    logger.info(f"Loading SFT dataset from {cfg.data.output_sft_file}...")
    dataset = load_dataset("json", data_files=cfg.data.output_sft_file)["train"]
    
    # Format text for training
    def format_chat(example):
        messages = [
            {"role": "system", "content": example["instruction"]},
            {"role": "user", "content": example["input"]},
            {"role": "assistant", "content": (
                f"Action: {example['output']['action']}\\n"
                f"Reasoning: {example['output']['reasoning']}\\n"
                f"Facets: {example['output']['facets']}\\n"
                f"Response: {example['output']['response']}"
            )}
        ]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        return {"text": formatted}
        
    formatted_dataset = dataset.map(format_chat, remove_columns=dataset.column_names)
    
    training_args = TrainingArguments(
        output_dir=cfg.training.output_dir,
        per_device_train_batch_size=cfg.training.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        num_train_epochs=cfg.training.num_train_epochs,
        learning_rate=cfg.training.learning_rate,
        warmup_ratio=cfg.training.warmup_ratio,
        bf16=cfg.training.bf16,
        logging_steps=cfg.training.logging_steps,
        save_steps=cfg.training.save_steps,
        save_total_limit=cfg.training.save_total_limit,
        optim=cfg.training.optim,
        report_to=cfg.training.report_to,
    )
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=formatted_dataset,
        dataset_text_field="text",
        max_seq_length=cfg.training.max_seq_length,
        packing=cfg.training.packing,
        args=training_args,
    )
    
    logger.info("Starting SFT Training...")
    trainer.train()
    
    trainer.save_model(f"{cfg.training.output_dir}/final")
    tokenizer.save_pretrained(f"{cfg.training.output_dir}/final")
    logger.info("SFT Training complete and model saved.")

def run_dpo_training(cfg: DictConfig):
    """Run Direct Preference Optimization."""
    logger.info("Initializing DPO Training...")
    
    # Load model and reference model
    model, tokenizer = load_model_and_tokenizer(cfg.model, is_train=True)
    ref_model, _ = load_model_and_tokenizer(cfg.model, is_train=False) # Ref model without LoRA adapters trainable
    
    logger.info(f"Loading DPO dataset from {cfg.data.output_dpo_file}...")
    dataset = load_dataset("json", data_files=cfg.data.output_dpo_file)["train"]
    
    training_args = TrainingArguments(
        output_dir=cfg.training.output_dir,
        per_device_train_batch_size=cfg.training.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        num_train_epochs=cfg.training.num_train_epochs,
        learning_rate=cfg.training.learning_rate,
        warmup_ratio=cfg.training.warmup_ratio,
        bf16=cfg.training.bf16,
        logging_steps=cfg.training.logging_steps,
        save_steps=cfg.training.save_steps,
        save_total_limit=cfg.training.save_total_limit,
        optim=cfg.training.optim,
        report_to=cfg.training.report_to,
    )
    
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        beta=cfg.training.beta,
        train_dataset=dataset,
        tokenizer=tokenizer,
        max_prompt_length=cfg.training.max_prompt_length,
        max_length=cfg.training.max_length,
    )
    
    logger.info("Starting DPO Training...")
    trainer.train()
    
    trainer.save_model(f"{cfg.training.output_dir}/final")
    tokenizer.save_pretrained(f"{cfg.training.output_dir}/final")
    logger.info("DPO Training complete and model saved.")
