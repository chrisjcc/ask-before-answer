---
title: Ask Before Answer Demo
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 8501
---

# AskBeforeAnswer 🤖

> A production-grade, clarification-seeking language model demonstration.

When a question is ambiguous (e.g., "How do I make pasta?"), multiple valid interpretations exist. Instead of hallucinating or guessing the user's intent, **AskBeforeAnswer** detects the ambiguity, explains its reasoning, identifies missing facets, and asks a targeted clarification question.

## 🚀 Application Details
This Hugging Face Space runs a custom Streamlit UI deployed via a Docker container. It queries the fine-tuned Qwen 2.5 7B underlying model.

### Features
- **Ambiguity Detection:** Identifies if a query requires clarification.
- **Facet Generation:** Extracts the missing pieces of information that would clarify the question.
- **Clarification:** Generates a human-like clarifying question to disambiguate the intent.
- **Direct Answer:** If the question is already specific enough, it answers directly!

For more information, research methodology, and training scripts, please refer to the main repository.
