"""Training pipelines for SFT and DPO.

This module contains the setup and execution logic for fine-tuning language models
using Supervised Fine-Tuning and Direct Preference Optimization via the TRL library.
"""

import ast
import logging
import os
import re
from typing import Any, Tuple

try:
    import torch._inductor.config  # noqa: F401
    import unsloth  # noqa: F401
except ImportError:
    pass

import torch
from datasets import load_dataset
from omegaconf import DictConfig
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.trainer_utils import get_last_checkpoint

logger = logging.getLogger(__name__)


def load_model_and_tokenizer(
    model_cfg: DictConfig, is_train: bool = True
) -> Tuple[Any, Any]:
    """Load tokenizer and model with given configuration.

    Args:
        model_cfg (DictConfig): The Hydra configuration for the model.
        is_train (bool): Whether to prepare the model for training
            (e.g. enable gradients).

    Returns:
        Tuple[Any, Any]: A tuple containing the loaded (model, tokenizer).
    """
    logger.info(f"Loading model {model_cfg.name}...")

    # Optional Unsloth integration for high-performance memory-efficient training
    use_unsloth = model_cfg.get("use_unsloth", False)
    load_in_8bit = model_cfg.get("load_in_8bit", False)
    load_in_4bit = model_cfg.get("load_in_4bit", False)

    if use_unsloth:
        try:
            from unsloth import FastLanguageModel

            logger.info("Unsloth is enabled. Loading via FastLanguageModel...")
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_cfg.name,
                max_seq_length=model_cfg.get("max_seq_length", 2048),
                dtype=(
                    getattr(torch, model_cfg.torch_dtype)
                    if hasattr(model_cfg, "torch_dtype")
                    else None
                ),
                load_in_4bit=load_in_4bit,
                trust_remote_code=model_cfg.get("trust_remote_code", False),
            )

            if is_train and "lora" in model_cfg:
                lora_cfg = model_cfg.lora
                logger.info("Applying Unsloth optimized LoRA adapters...")
                model = FastLanguageModel.get_peft_model(
                    model,
                    r=lora_cfg.r,
                    target_modules=list(lora_cfg.target_modules),
                    lora_alpha=lora_cfg.lora_alpha,
                    lora_dropout=lora_cfg.lora_dropout,
                    bias=lora_cfg.bias,
                    use_gradient_checkpointing="unsloth",
                    random_state=3407,
                )

            return model, tokenizer

        except ImportError as e:
            logger.warning(
                f"unsloth import failed: {e}\n"
                "Falling back to native HuggingFace transformers."
            )

    # Native HuggingFace Fallback / Standard loading
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
                logger.info("Applied native LoRA configuration.")

    return model, tokenizer


def run_sft_training(cfg: DictConfig):
    """Run Supervised Fine-Tuning."""
    logger.info("Initializing SFT Training...")

    model, tokenizer = load_model_and_tokenizer(cfg.model, is_train=True)
    from datasets import Features, Sequence, Value
    from trl import SFTConfig, SFTTrainer

    sft_features = Features(
        {
            "instruction": Value("string"),
            "input": Value("string"),
            "output": Features(
                {
                    "action": Value("string"),
                    "reasoning": Value("string"),
                    "facets": Sequence(Value("string")),
                    "response": Value("string"),
                }
            ),
        }
    )

    logger.info(
        f"Loading SFT training dataset from {cfg.data.output_sft_train_file}..."
    )
    dataset_train = load_dataset(
        "json", data_files=cfg.data.output_sft_train_file, features=sft_features
    )["train"]

    logger.info(
        f"Loading SFT validation dataset from {cfg.data.output_sft_val_file}..."
    )
    dataset_val = load_dataset(
        "json", data_files=cfg.data.output_sft_val_file, features=sft_features
    )["train"]

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
        max_steps=cfg.training.get("max_steps", -1),
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
        run_name=os.environ.get("WANDB_RUN_ID", "sft_training"),
        dataset_text_field="text",
        max_length=cfg.training.max_seq_length,
        packing=cfg.training.packing,
        remove_unused_columns=cfg.training.get("remove_unused_columns", True),
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=formatted_train,
        eval_dataset=formatted_val,
        args=training_args,
    )

    logger.info("Starting SFT Training...")
    last_checkpoint = None
    if cfg.training.get("resume_from_checkpoint", False) and os.path.isdir(
        cfg.training.output_dir
    ):
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

    from trl import DPOConfig, DPOTrainer

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

    # Wrap DPO dataset with ChatML to match SFT
    def format_dpo(example):
        system_prompt = (
            "You are a helpful assistant. "
            "Given a question, you must decide whether it is ambiguous or not. "
            "Output MUST follow this format:\n"
            "Action: Clarify|Answer\n"
            "Reasoning: <your reasoning>\n"
            "Facets: <list of facets if ambiguous, else empty>\n"
            "Response: <clarifying question or direct answer>"
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": example["prompt"]},
        ]

        prompt_str = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        return {
            "prompt": prompt_str,
            "chosen": example["chosen"] + tokenizer.eos_token,
            "rejected": example["rejected"] + tokenizer.eos_token,
        }

    dataset_train = dataset_train.map(format_dpo)
    dataset_val = dataset_val.map(format_dpo)

    training_args = DPOConfig(
        output_dir=cfg.training.output_dir,
        per_device_train_batch_size=cfg.training.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        num_train_epochs=cfg.training.num_train_epochs,
        max_steps=cfg.training.get("max_steps", -1),
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
        run_name=os.environ.get("WANDB_RUN_ID", "dpo_training"),
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
        processing_class=tokenizer,
    )

    logger.info("Starting DPO Training...")
    last_checkpoint = None
    if cfg.training.get("resume_from_checkpoint", False) and os.path.isdir(
        cfg.training.output_dir
    ):
        last_checkpoint = get_last_checkpoint(cfg.training.output_dir)
        if last_checkpoint is not None:
            logger.info(f"Resuming DPO training from {last_checkpoint}")

    trainer.train(resume_from_checkpoint=last_checkpoint)

    trainer.save_model(f"{cfg.training.output_dir}/final")
    tokenizer.save_pretrained(f"{cfg.training.output_dir}/final")
    logger.info("DPO Training complete and model saved.")


def run_orpo_training(cfg: DictConfig):
    """Run Odds Ratio Preference Optimization."""
    logger.info("Initializing ORPO Training...")

    # Load model (NO reference model needed for ORPO)
    model, tokenizer = load_model_and_tokenizer(cfg.model, is_train=True)
    from trl import ORPOConfig, ORPOTrainer

    logger.info(
        f"Loading ORPO training dataset from {cfg.data.output_dpo_train_file}..."
    )
    dataset_train = load_dataset("json", data_files=cfg.data.output_dpo_train_file)[
        "train"
    ]

    logger.info(
        f"Loading ORPO validation dataset from {cfg.data.output_dpo_val_file}..."
    )
    dataset_val = load_dataset("json", data_files=cfg.data.output_dpo_val_file)["train"]

    # Wrap DPO dataset with ChatML to match SFT
    def format_dpo(example):
        system_prompt = (
            "You are a helpful assistant. "
            "Given a question, you must decide whether it is ambiguous or not. "
            "Output MUST follow this format:\n"
            "Action: Clarify|Answer\n"
            "Reasoning: <your reasoning>\n"
            "Facets: <list of facets if ambiguous, else empty>\n"
            "Response: <clarifying question or direct answer>"
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": example["prompt"]},
        ]

        prompt_str = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        return {
            "prompt": prompt_str,
            "chosen": example["chosen"] + tokenizer.eos_token,
            "rejected": example["rejected"] + tokenizer.eos_token,
        }

    dataset_train = dataset_train.map(format_dpo)
    dataset_val = dataset_val.map(format_dpo)

    training_args = ORPOConfig(
        output_dir=cfg.training.output_dir,
        per_device_train_batch_size=cfg.training.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        num_train_epochs=cfg.training.num_train_epochs,
        max_steps=cfg.training.get("max_steps", -1),
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
        run_name=os.environ.get("WANDB_RUN_ID", "orpo_training"),
        beta=cfg.training.beta,
        max_prompt_length=cfg.training.max_prompt_length,
        max_length=cfg.training.max_length,
        remove_unused_columns=cfg.training.get("remove_unused_columns", False),
    )

    trainer = ORPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset_train,
        eval_dataset=dataset_val,
        processing_class=tokenizer,
    )

    logger.info("Starting ORPO Training...")
    last_checkpoint = None
    if cfg.training.get("resume_from_checkpoint", False) and os.path.isdir(
        cfg.training.output_dir
    ):
        last_checkpoint = get_last_checkpoint(cfg.training.output_dir)
        if last_checkpoint is not None:
            logger.info(f"Resuming ORPO training from {last_checkpoint}")

    trainer.train(resume_from_checkpoint=last_checkpoint)

    trainer.save_model(f"{cfg.training.output_dir}/final")
    tokenizer.save_pretrained(f"{cfg.training.output_dir}/final")
    logger.info("ORPO Training complete and model saved.")


def format_reward_func(prompts, completions, reward_weights=None, **kwargs):
    """Reward function that checks for the exact format constraints."""
    if reward_weights is None:
        reward_weights = {}
    format_reward = reward_weights.get("format_reward", 1.0)
    format_penalty = reward_weights.get("format_penalty", -2.0)

    rewards = []
    for completion in completions:
        # A simple check: do we have all the sections in order?
        text = completion[0]["content"] if isinstance(completion, list) else completion
        has_action = "Action:" in text
        has_reasoning = "Reasoning:" in text
        has_facets = "Facets:" in text
        has_response = "Response:" in text

        if has_action and has_reasoning and has_facets and has_response:
            rewards.append(format_reward)
        else:
            rewards.append(format_penalty)
    return rewards


def action_reward_func(prompts, completions, reward_weights=None, **kwargs):
    """Reward function that checks if the predicted action matches the target."""
    if reward_weights is None:
        reward_weights = {}
    action_reward = reward_weights.get("action_reward", 1.0)
    action_penalty = reward_weights.get("action_penalty", -1.0)

    rewards = []
    target_actions = kwargs.get("target_action", [])

    for i, completion in enumerate(completions):
        text = completion[0]["content"] if isinstance(completion, list) else completion
        target = target_actions[i]

        action_match = re.search(r"Action:\s*(Clarify|Answer)", text)
        if action_match:
            pred_action = action_match.group(1)
            if pred_action == target:
                rewards.append(action_reward)
            else:
                rewards.append(action_penalty)
        else:
            rewards.append(action_penalty)
    return rewards


def facet_logic_reward_func(prompts, completions, reward_weights=None, **kwargs):
    """Reward function that checks facet presence/absence based on action."""
    if reward_weights is None:
        reward_weights = {}
    facet_reward = reward_weights.get("facet_logic_reward", 0.5)
    facet_penalty = reward_weights.get("facet_logic_penalty", -0.5)

    rewards = []
    for completion in completions:
        text = completion[0]["content"] if isinstance(completion, list) else completion
        action_match = re.search(r"Action:\s*(Clarify|Answer)", text)
        facets_match = re.search(r"Facets:\s*(\[.*?\])", text, re.DOTALL)

        if not action_match or not facets_match:
            rewards.append(facet_penalty)
            continue

        pred_action = action_match.group(1)
        facets_str = facets_match.group(1)

        try:
            facets = ast.literal_eval(facets_str)
            if not isinstance(facets, list):
                facets = []
        except Exception:
            facets = []

        if pred_action == "Clarify":
            # Clarify MUST have non-empty facets
            if len(facets) > 0:
                rewards.append(facet_reward)
            else:
                rewards.append(facet_penalty)
        else:
            # Answer MUST have empty facets
            if len(facets) == 0:
                rewards.append(facet_reward)
            else:
                rewards.append(facet_penalty)
    return rewards


def accuracy_reward_func(prompts, completions, reward_weights=None, **kwargs):
    """Reward function that checks for factual accuracy (word overlap)
    for direct answers."""
    if reward_weights is None:
        reward_weights = {}
    acc_scale = reward_weights.get("accuracy_scale", 1.5)
    acc_shift = reward_weights.get("accuracy_shift", -0.5)
    acc_miss_penalty = reward_weights.get("accuracy_miss_penalty", -1.0)
    acc_format_penalty = reward_weights.get("accuracy_format_penalty", -0.5)

    rewards = []
    target_responses = kwargs.get("target_response", [])
    target_actions = kwargs.get("target_action", [])

    for i, completion in enumerate(completions):
        text = completion[0]["content"] if isinstance(completion, list) else completion
        target_resp = target_responses[i]
        target_act = target_actions[i]

        # Only heavily shape accuracy for direct answers to prevent hallucination.
        # Clarification questions are too linguistically diverse to strictly grade
        # with word overlap.
        if target_act != "Answer":
            rewards.append(0.0)
            continue

        response_match = re.search(r"Response:\s*(.*)", text, re.DOTALL)
        if not response_match or not target_resp:
            rewards.append(acc_format_penalty)
            continue

        pred_resp = response_match.group(1).strip()

        # Token overlap F1
        pred_words = set(re.findall(r"\b\w+\b", pred_resp.lower()))
        target_words = set(re.findall(r"\b\w+\b", target_resp.lower()))

        if not target_words:
            rewards.append(0.0)
            continue

        intersection = pred_words.intersection(target_words)

        if not intersection:
            rewards.append(acc_miss_penalty)  # Totally missed the facts
            continue

        recall = len(intersection) / len(target_words)
        precision = len(intersection) / len(pred_words)

        f1 = 2 * (precision * recall) / (precision + recall)

        # Map F1 to reward: low overlap is penalized,
        # high overlap is moderately rewarded
        reward = (f1 * acc_scale) + acc_shift
        rewards.append(reward)

    return rewards


def run_grpo_training(cfg: DictConfig):
    """Run Group Relative Policy Optimization."""
    logger.info("Initializing GRPO Training...")

    model, tokenizer = load_model_and_tokenizer(cfg.model, is_train=True)
    from trl import GRPOConfig, GRPOTrainer

    logger.info(
        f"Loading GRPO training dataset from {cfg.data.output_dpo_train_file}..."
    )
    dataset_train = load_dataset("json", data_files=cfg.data.output_dpo_train_file)[
        "train"
    ]
    logger.info(
        f"Loading GRPO validation dataset from {cfg.data.output_dpo_val_file}..."
    )
    dataset_val = load_dataset("json", data_files=cfg.data.output_dpo_val_file)["train"]

    def format_grpo(example):
        system_prompt = (
            "You are a helpful assistant. "
            "Given a question, you must decide whether it is ambiguous or not. "
            "Output MUST follow this format:\n"
            "Action: Clarify|Answer\n"
            "Reasoning: <your reasoning>\n"
            "Facets: <list of facets if ambiguous, else empty>\n"
            "Response: <clarifying question or direct answer>"
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": example["prompt"]},
        ]

        prompt_str = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Extract the target action from the 'chosen' string in the DPO dataset
        target_action = "Answer"
        if "Action: Clarify" in example["chosen"]:
            target_action = "Clarify"

        target_response = ""
        response_match = re.search(r"Response:\s*(.*)", example["chosen"], re.DOTALL)
        if response_match:
            target_response = response_match.group(1).strip()

        return {
            "prompt": prompt_str,
            "target_action": target_action,
            "target_response": target_response,
        }

    dataset_train = dataset_train.map(
        format_grpo, remove_columns=dataset_train.column_names
    )
    dataset_val = dataset_val.map(format_grpo, remove_columns=dataset_val.column_names)

    training_args = GRPOConfig(
        output_dir=cfg.training.output_dir,
        per_device_train_batch_size=cfg.training.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        num_train_epochs=cfg.training.num_train_epochs,
        max_steps=cfg.training.get("max_steps", -1),
        learning_rate=cfg.training.learning_rate,
        warmup_ratio=cfg.training.warmup_ratio,
        bf16=cfg.training.bf16,
        logging_steps=cfg.training.logging_steps,
        save_steps=cfg.training.save_steps,
        save_total_limit=cfg.training.save_total_limit,
        optim=cfg.training.optim,
        report_to=cfg.training.report_to,
        run_name=os.environ.get("WANDB_RUN_ID", "grpo_training"),
        beta=cfg.training.beta,
        max_prompt_length=cfg.training.max_prompt_length,
        max_completion_length=cfg.training.max_completion_length,
        num_generations=cfg.training.num_generations,
        loss_type=cfg.training.get("loss_type", "grpo"),
    )

    # Bugfix for UnslothGRPOTrainer CPU/CUDA device mismatch during generation
    original_generate = model.generate

    def patched_generate(*args, **kwargs):
        if len(args) > 0 and isinstance(args[0], torch.Tensor):
            args = list(args)
            args[0] = args[0].to(model.device)
            args = tuple(args)
        if "input_ids" in kwargs and isinstance(kwargs["input_ids"], torch.Tensor):
            kwargs["input_ids"] = kwargs["input_ids"].to(model.device)
        if "attention_mask" in kwargs and isinstance(
            kwargs["attention_mask"], torch.Tensor
        ):
            kwargs["attention_mask"] = kwargs["attention_mask"].to(model.device)
        return original_generate(*args, **kwargs)

    model.generate = patched_generate

    from functools import partial

    reward_weights = cfg.training.get("reward_weights", {})

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[
            partial(format_reward_func, reward_weights=reward_weights),
            partial(action_reward_func, reward_weights=reward_weights),
            partial(facet_logic_reward_func, reward_weights=reward_weights),
            partial(accuracy_reward_func, reward_weights=reward_weights),
        ],
        args=training_args,
        train_dataset=dataset_train,
        eval_dataset=dataset_val,
        processing_class=tokenizer,
    )

    logger.info("Starting GRPO Training...")
    last_checkpoint = None
    if cfg.training.get("resume_from_checkpoint", False) and os.path.isdir(
        cfg.training.output_dir
    ):
        last_checkpoint = get_last_checkpoint(cfg.training.output_dir)
        if last_checkpoint is not None:
            logger.info(f"Resuming GRPO training from {last_checkpoint}")

    trainer.train(resume_from_checkpoint=last_checkpoint)

    trainer.save_model(f"{cfg.training.output_dir}/final")
    tokenizer.save_pretrained(f"{cfg.training.output_dir}/final")
    logger.info("GRPO Training complete and model saved.")
