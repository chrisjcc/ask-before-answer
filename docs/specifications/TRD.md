# Technical Requirements Document (TRD)
**Project Name:** AskBeforeAnswer
**Date:** September 2026
**Audience:** Stakeholders, Product Owners, and Engineering Teams

## 1. Problem Statement & Motivation
Open-domain Large Language Models (LLMs) often hallucinate or guess user intent when faced with ambiguous queries (e.g., "How do I make pasta?"). Instead of confidently answering with assumptions, an intelligent assistant should actively seek clarification. The goal of the AskBeforeAnswer project is to train an LLM to autonomously detect ambiguity, extract missing semantic facets, and ask targeted clarifying questions before answering.

## 2. Scope Boundaries
**In-Scope:**
- Single-turn analysis (query routing to either Clarify or Answer).
- Training an open-weight foundation model (Qwen 2.5 7B) using post-training alignment techniques.
- Synthetic data generation for contrastive preference pairs.
- Real-time user inference via a web UI.

**Out-of-Scope:**
- Multi-turn dialogue management (caching context across multiple subsequent interactions).
- Web search or external RAG retrieval integration.
- Foundational pre-training (training from scratch).

## 3. Functional Requirements
1. **Ambiguity Detection:** The system must accurately classify incoming user queries as either ambiguous or clear.
2. **Schema Adherence:** The model's output must perfectly adhere to a structured 4-field schema:
   - `Action:` Must be exactly "Clarify" or "Answer".
   - `Reasoning:` A Chain-of-Thought (CoT) trace explaining the action logic.
   - `Facets:` A valid JSON list of missing information (if Clarify) or an empty list (if Answer).
   - `Response:` The final conversational question or direct answer.
3. **Structured Disambiguation:** When asking a clarifying question, the system must explicitly ground its question in the extracted `Facets`.
4. **Factual Answering:** When a question is clear, the system must provide a factually accurate direct answer without hallucinating.

## 4. Non-Functional Requirements & Constraints
1. **Model Size:** The base model must not exceed 7 Billion parameters (e.g., Qwen 2.5 7B) to ensure rapid inference.
2. **Memory Footprint:** The final model must be deployable on edge/consumer hardware, fitting within a minimum of 8-12GB VRAM (using 4-bit/8-bit precision) for inference.
3. **Latency:** End-to-end routing and generation should execute in under 3 seconds per query on GPU hardware (e.g., NVIDIA L4).
4. **Observability:** All inference traces and model predictions must be logged in real-time to a central telemetry dashboard for compliance and evaluation.

## 5. Acceptance Criteria
The project is considered successful when the following metric thresholds are achieved on the `sewon_ambig_qa_eval` benchmark dataset:
- **Macro F1 Score:** > 0.60 (demonstrating balanced calibration between clarification and answering).
- **Ambiguity Detection Accuracy:** > 0.95 (reliably identifying ambiguous queries).
- **Facet Generation Rate:** 1.0 (the model must *never* ask a clarifying question without first extracting a facet).
- **Usefulness Score:** > 0.85 as determined by a stochastic LLM-as-a-Judge (e.g., Gemini 2.5 Flash).
