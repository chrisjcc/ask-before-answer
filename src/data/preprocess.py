"""Data preprocessing pipeline for AskBeforeAnswer.

This module is responsible for loading the raw datasets and processing them
into the structured JSONL formats required for Supervised Fine-Tuning (SFT)
and Direct Preference Optimization (DPO).
"""

import ast
import json
import logging
import re
from typing import Any, Dict, List, Optional

import pandas as pd
from datasets import load_dataset
from omegaconf import DictConfig
from tqdm import tqdm

logger = logging.getLogger(__name__)


def clean_facets(facets: Any) -> List[str]:
    """Clean and parse facet lists."""
    if isinstance(facets, list):
        return [str(x) for x in facets]
    if isinstance(facets, str):
        try:
            val = ast.literal_eval(facets)
            return [str(x) for x in val] if isinstance(val, list) else []
        except Exception:
            return []
    return []


def clean_response(resp: Any) -> str:
    """Clean model responses."""
    if resp is None:
        return ""
    if isinstance(resp, list):
        return str(resp[0]) if resp else ""
    if isinstance(resp, str) and resp.startswith("[") and resp.endswith("]"):
        try:
            lst = ast.literal_eval(resp)
            return str(lst[0]) if lst else ""
        except Exception:
            pass
    return str(resp).strip()


class SyntheticGenerator:
    """Generates synthetic reasoning, facets, and responses using a lightweight LLM."""

    SYSTEM_PROMPT = """You are a clarification-seeking question-understanding
agent trained on AmbigQA.

For each user question:
1) Decide if it is ambiguous (i.e., multiple plausible interpretations exist).
2) Explain your reasoning deeply.
3) List facets if ambiguous.
4) Ask a clarifying question OR give a direct answer.

Format MUST be exactly:
Action: Clarify|Answer
Reasoning: <detailed reasoning explaining why the question
has multiple interpretations, or why it is perfectly clear>
Facets: [list, of, facets]   # empty list [] if unambiguous
Response: <clarifying question OR direct answer>
"""

    def __init__(self, model_id: str, batch_size: int = 8, max_new_tokens: int = 256):
        from transformers import logging as tf_logging
        from unsloth import FastLanguageModel

        tf_logging.set_verbosity_error()

        logger.info(f"Loading synthetic generator model: {model_id} via Unsloth")

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_id,
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(self.model)

        # Handle pad token and left-padding for batched generation
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        self.model.eval()
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens

    def _parse_output(self, text: str) -> Dict[str, Any]:
        """Parses the generated text into structured fields."""
        action_match = re.search(r"Action:\s*(Clarify|Answer)", text, re.IGNORECASE)
        action = action_match.group(1).title() if action_match else "Answer"

        reasoning_match = re.search(
            r"Reasoning:\s*(.*?)(?=\nFacets:|\Z)", text, re.DOTALL | re.IGNORECASE
        )
        reasoning = (
            reasoning_match.group(1).strip()
            if reasoning_match
            else "The question is missing specific details."
        )

        facets_match = re.search(
            r"Facets:\s*(\[.*?\])", text, re.DOTALL | re.IGNORECASE
        )
        facets = []
        if facets_match:
            try:
                facets = ast.literal_eval(facets_match.group(1).strip())
            except (SyntaxError, ValueError):
                pass

        response_match = re.search(r"Response:\s*(.*)", text, re.DOTALL | re.IGNORECASE)
        response = response_match.group(1).strip() if response_match else ""

        return {
            "action": action,
            "reasoning": reasoning,
            "facets": facets if isinstance(facets, list) else [],
            "response": response,
        }

    def generate_batch(self, questions: List[str]) -> List[Dict[str, Any]]:
        """Generates structured outputs for a batch of questions."""
        import torch

        prompts = [
            self.tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": q},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            for q in questions
        ]

        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(
            self.model.device
        )

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        results = []
        for i, output in enumerate(outputs):
            input_len = inputs.input_ids[i].shape[0]
            generated_text = self.tokenizer.decode(
                output[input_len:], skip_special_tokens=True
            )
            results.append(self._parse_output(generated_text))

        return results


def extract_qa_data(
    dataset_name: str,
    split: str = "train",
    max_samples: Optional[int] = None,
    synthetic_cfg: Optional[DictConfig] = None,
) -> pd.DataFrame:
    """Load AmbigQA dataset and extract question and annotation details.

    Args:
        dataset_name (str): The name or path of the dataset to load from HuggingFace.
        split (str): The dataset split to process (e.g., 'train', 'validation').
        max_samples (Optional[int]): Maximum number of rows to extract.
        synthetic_cfg (Optional[DictConfig]): Configuration for LLM synthetic
            generation.

    Returns:
        pd.DataFrame: A DataFrame containing the extracted questions and facets.
    """
    logger.info(f"Loading dataset {dataset_name} ({split})...")
    ds = load_dataset(dataset_name, split=split)
    if max_samples:
        ds = ds.select(range(min(len(ds), max_samples)))

    rows = []
    questions = []

    # Base extraction
    for entry in ds:
        ann = entry["annotations"]
        ann_type = ann["type"][0] if isinstance(ann["type"], list) else ann["type"]
        qa_pairs = ann.get("qaPairs", [])
        if len(qa_pairs) == 1 and isinstance(qa_pairs[0], list):
            qa_pairs = qa_pairs[0]

        single_ans_list = ann.get("answer", [])
        single_ans_flat = []
        if isinstance(single_ans_list, list):
            for ans in single_ans_list:
                if isinstance(ans, list):
                    single_ans_flat.extend(ans)
                elif isinstance(ans, str):
                    single_ans_flat.append(ans)

        is_ambiguous = ann_type == "multipleQAs"
        questions.append(entry["question"])

        # Determine base truth for fallback and correctness checks
        base_action = "Clarify" if is_ambiguous else "Answer"
        base_pos_resp = (
            qa_pairs[0].get("question", "Could you clarify?")
            if is_ambiguous and qa_pairs
            else (single_ans_flat[0] if single_ans_flat else "Direct answer.")
        )

        rows.append(
            {
                "question": entry["question"],
                "is_ambiguous": is_ambiguous,
                "action": base_action,
                "facets": ["Entity Reference"] if is_ambiguous else [],
                "reasoning": (
                    "The question is missing specific details."
                    if is_ambiguous
                    else "The question is clear."
                ),
                "positive_response": base_pos_resp,
                "neg_response": (
                    single_ans_flat[0]
                    if is_ambiguous and single_ans_flat
                    else "Could you please clarify the specific details?"
                ),
            }
        )

    df = pd.DataFrame(rows)

    # Apply synthetic generation if enabled
    if synthetic_cfg and synthetic_cfg.get("enabled", False):
        generator = SyntheticGenerator(
            model_id=synthetic_cfg.get("model_id", "Qwen/Qwen2.5-3B-Instruct"),
            batch_size=synthetic_cfg.get("batch_size", 8),
            max_new_tokens=synthetic_cfg.get("max_new_tokens", 256),
        )

        logger.info(
            f"Generating synthetic annotations for {len(questions)} examples..."
        )
        batch_size = generator.batch_size

        for i in tqdm(range(0, len(questions), batch_size), desc="Synthetic Gen"):
            batch_q = questions[i : i + batch_size]
            batch_results = generator.generate_batch(batch_q)

            for j, res in enumerate(batch_results):
                idx = i + j
                # Overwrite placeholder fields with LLM-generated high-quality data
                # We enforce the ground-truth action from AmbigNQ
                # to ensure correct labels
                is_amb = df.at[idx, "is_ambiguous"]

                df.at[idx, "reasoning"] = res["reasoning"]
                df.at[idx, "facets"] = res["facets"] if is_amb else []

                # If the LLM successfully generated a response that matches
                # the target action, use it
                # Otherwise, fallback to the dataset's ground truth
                # to avoid noisy positives
                if res["action"] == df.at[idx, "action"] and res["response"]:
                    df.at[idx, "positive_response"] = res["response"]
                else:
                    if synthetic_cfg.get("filter_bad_rows", False):
                        df.at[idx, "is_bad_row"] = True

                # Extra validation for facets
                if synthetic_cfg.get("filter_bad_rows", False):
                    if is_amb and not res["facets"]:
                        df.at[idx, "is_bad_row"] = True

        if synthetic_cfg.get("filter_bad_rows", False):
            if "is_bad_row" in df.columns:
                num_bad = df["is_bad_row"].sum()
                logger.info(f"Dropping {num_bad} bad rows generated by LLM.")
                df = df[~df["is_bad_row"]].reset_index(drop=True)
                df = df.drop(columns=["is_bad_row"])

        if synthetic_cfg.get("balance_classes", False):
            clarify_df = df[df["action"] == "Clarify"]
            answer_df = df[df["action"] == "Answer"]
            min_len = min(len(clarify_df), len(answer_df))
            if min_len > 0:
                logger.info(f"Balancing classes to {min_len} samples each.")
                clarify_df = clarify_df.sample(n=min_len, random_state=42)
                answer_df = answer_df.sample(n=min_len, random_state=42)
                df = (
                    pd.concat([clarify_df, answer_df])
                    .sample(frac=1, random_state=42)
                    .reset_index(drop=True)
                )

    return df


def prepare_sft_dataset(df: pd.DataFrame, output_path: str) -> None:
    """Format DataFrame into SFT JSONL format and save to disk."""
    logger.info(f"Preparing SFT dataset to {output_path}...")
    records = []

    system_instruction = (
        "You are a helpful assistant. "
        "Given a question, you must decide whether it is ambiguous or not. "
        "Output MUST follow this format:\n"
        "Action: Clarify|Answer\n"
        "Reasoning: <your reasoning>\n"
        "Facets: <list of facets if ambiguous, else empty>\n"
        "Response: <clarifying question or direct answer>"
    )

    for _, row in df.iterrows():
        is_amb = row["is_ambiguous"]
        facets = clean_facets(row["facets"])

        records.append(
            {
                "instruction": system_instruction,
                "input": row["question"],
                "output": {
                    "action": row["action"],
                    "reasoning": row["reasoning"],
                    "facets": facets if is_amb else [],
                    "response": clean_response(row["positive_response"]),
                },
            }
        )

    with open(output_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    logger.info(f"Saved {len(records)} records for SFT.")


def prepare_dpo_dataset(
    df: pd.DataFrame,
    output_path: str,
    synthetic_cfg: Optional[DictConfig] = None
) -> None:
    """Format DataFrame into DPO JSONL format and save to disk."""
    logger.info(f"Preparing DPO dataset to {output_path}...")
    records = []

    for _, row in df.iterrows():
        # DPO requires chosen vs rejected
        action = row["action"]
        reasoning = row["reasoning"]

        is_amb = row["is_ambiguous"]
        facets_list = clean_facets(row["facets"]) if is_amb else []
        facets_str = str(facets_list)

        chosen_resp = clean_response(row["positive_response"])
        chosen = (
            f"Action: {action}\n"
            f"Reasoning: {reasoning}\n"
            f"Facets: {facets_str}\n"
            f"Response: {chosen_resp}"
        )

        generate_hard_negatives = synthetic_cfg and synthetic_cfg.get(
            "generate_hard_negatives", False
        )

        if generate_hard_negatives:
            neg_resp = clean_response(
                row.get("neg_response", "Could you please clarify?")
            )
            if action == "Clarify":
                rejected_action = "Answer"
                rejected = (
                    f"Action: {rejected_action}\n"
                    f"Reasoning: {reasoning}\n"
                    f"Facets: []\n"
                    f"Response: {neg_resp}"
                )
            else:
                rejected_action = "Clarify"
                rejected = (
                    f"Action: {rejected_action}\n"
                    f"Reasoning: {reasoning}\n"
                    f"Facets: ['General Context']\n"
                    f"Response: {neg_resp}"
                )
        else:
            rejected_action = "Answer" if action == "Clarify" else "Clarify"
            rejected = (
                f"Action: {rejected_action}\n"
               "Reasoning: Incorrect reasoning: the model misunderstood "
               "the question.\n"
              'Facets: ["Incorrect Interpretation"]\n'
              "Response: "
              + ("I don't know." if action == "Clarify" else "Could you clarify?")
            )

        records.append(
            {
                "prompt": row["question"],
                "chosen": chosen,
                "rejected": rejected,
            }
        )

    with open(output_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    logger.info(f"Saved {len(records)} records for DPO.")
