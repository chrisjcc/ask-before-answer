"""Data preprocessing pipeline for AskBeforeAnswer.

This module is responsible for loading the raw datasets and processing them
into the structured JSONL formats required for Supervised Fine-Tuning (SFT)
and Direct Preference Optimization (DPO).
"""

import ast
import json
import logging
from typing import Any, List, Optional

import pandas as pd
from datasets import load_dataset

logger = logging.getLogger(__name__)


def clean_facets(facets: Any) -> List[str]:
    """Clean and parse facet lists."""
    if isinstance(facets, list):
        return facets
    if isinstance(facets, str):
        try:
            val = ast.literal_eval(facets)
            return val if isinstance(val, list) else []
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


def extract_qa_data(
    dataset_name: str, split: str = "train", max_samples: Optional[int] = None
) -> pd.DataFrame:
    """Load AmbigQA dataset and extract question and annotation details.

    Args:
        dataset_name (str): The name or path of the dataset to load from HuggingFace.
        split (str): The dataset split to process (e.g., 'train', 'validation').
        max_samples (Optional[int]): Maximum number of rows to extract.

    Returns:
        pd.DataFrame: A DataFrame containing the extracted questions and facets.
    """
    logger.info(f"Loading dataset {dataset_name} ({split})...")
    ds = load_dataset(dataset_name, split=split)
    if max_samples:
        ds = ds.select(range(min(len(ds), max_samples)))

    rows = []
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

        rows.append(
            {
                "question": entry["question"],
                "is_ambiguous": is_ambiguous,
                "action": "Clarify" if is_ambiguous else "Answer",
                # We would normally generate facets and reasoning via an LLM
                # or use ground truth. For this pipeline template, we will
                # use mock/extracted values.
                "facets": ["Entity Reference"] if is_ambiguous else [],
                "reasoning": (
                    "The question is missing specific details."
                    if is_ambiguous
                    else "The question is clear."
                ),
                "positive_response": (
                    qa_pairs[0].get("question", "Could you clarify?")
                    if is_ambiguous and qa_pairs
                    else "Direct answer."
                ),
            }
        )

    df = pd.DataFrame(rows)
    return df


def prepare_sft_dataset(df: pd.DataFrame, output_path: str) -> None:
    """Format DataFrame into SFT JSONL format and save to disk.

    Args:
        df (pd.DataFrame): The extracted dataset.
        output_path (str): The file path where the JSONL should be written.
    """
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


def prepare_dpo_dataset(df: pd.DataFrame, output_path: str) -> None:
    """Format DataFrame into DPO JSONL format and save to disk.

    Args:
        df (pd.DataFrame): The extracted dataset.
        output_path (str): The file path where the JSONL should be written.
    """
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
        rejected_action = "Answer" if action == "Clarify" else "Clarify"
        rejected = (
            f"Action: {rejected_action}\n"
            "Reasoning: I am not sure.\n"
            "Facets: []\n"
            "Response: I don't know."
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
