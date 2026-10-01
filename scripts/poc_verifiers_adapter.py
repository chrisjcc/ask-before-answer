#!/usr/bin/env python
"""
Proof of Concept: Verifiers Async-to-Sync Adapter
Tests the latency of running verifiers.v1.Judge concurrently across a batch of completions.

This mimics what would happen inside TRL's make_reward_functions during GRPO training.
"""

import asyncio
import time
import os
import argparse
from typing import List, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# To test this, you must have the `verifiers` package installed:
# pip install git+https://github.com/PrimeIntellect-ai/verifiers.git
try:
    import verifiers.v1 as vf
    from verifiers.v1.trace import Trace, TraceTask, AgentInfo
    from verifiers.v1.configs.judge import JudgeConfig
    from verifiers.v1.configs.agent import AgentConfig
except ImportError:
    logger.error("The 'verifiers' library is not installed. Please install it to run this PoC.")
    exit(1)


class SemanticUsefulnessJudge(vf.Judge[float]):
    """
    A Judge that evaluates whether a clarification question is semantically useful.
    Returns a float score between -1.0 and 1.0.
    """
    prompt = (
        "You are an expert evaluator for an AI assistant. "
        "The model generated the following response to an ambiguous query.\n\n"
        "Original Prompt: {prompt}\n"
        "Model Response: {response}\n\n"
        "Evaluate the usefulness of the response. If it's a clarification question, "
        "does it ask for the necessary missing information? \n"
        "Return a score between -1.0 (bad/hallucinated) and 1.0 (excellent/helpful)."
    )

    def parse(self, response: vf.JudgeResponse[float]) -> float:
        text = response.text.strip()
        try:
            # Simple float extraction from the response text
            import re
            match = re.search(r"[-+]?\d*\.\d+|\d+", text)
            if match:
                score = float(match.group())
                return max(-1.0, min(1.0, score))
            return 0.0
        except Exception:
            return 0.0


async def _score_single_trace(judge: SemanticUsefulnessJudge, prompt: str, completion: str) -> float:
    """Helper to score a single prompt/completion pair."""
    # Create a mock trace (the bare minimum required by the Judge)
    trace = Trace(
        task=TraceTask(type="AmbigQATask", data={}, key="mock", hash="mock"),
        state={},
        agent=AgentInfo(config=AgentConfig())
    )
    # Typically trace.transcript / trace.last_reply would be populated, but 
    # we explicitly pass the prompt and response to the judge here.
    try:
        result = await judge.evaluate(
            trace=trace,
            prompt=prompt,
            response=completion,
        )
        return float(result.parsed) if result.parsed is not None else 0.0
    except Exception as e:
        logger.error(f"Error scoring trace: {e}")
        return 0.0


def verifiers_reward_adapter(prompts: List[str], completions: List[str], judge_model: str = "gemini-2.5-flash") -> List[float]:
    """
    The synchronous adapter function that TRL would call.
    """
    judge_config = JudgeConfig(
        model=judge_model,
        # Increase concurrency limits or timeout settings here if needed
    )
    judge = SemanticUsefulnessJudge(judge_config)
    
    start_time = time.time()
    
    # Spin up an event loop to handle the async API calls concurrently
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        tasks = [
            _score_single_trace(judge, p, c) 
            for p, c in zip(prompts, completions)
        ]
        
        # Gather all scores concurrently
        rewards = loop.run_until_complete(asyncio.gather(*tasks))
        
    finally:
        loop.close()
        
    end_time = time.time()
    
    batch_size = len(prompts)
    duration = end_time - start_time
    logger.info(f"Processed batch of {batch_size} in {duration:.2f} seconds ({duration/batch_size:.2f}s per item).")
    
    return rewards


def main():
    parser = argparse.ArgumentParser(description="Test PrimeIntellect Verifiers adapter latency.")
    parser.add_argument("--batch-size", type=int, default=16, help="Simulate a TRL batch size.")
    parser.add_argument("--model", type=str, default="gemini-2.5-flash", help="Judge model to use (e.g. gemini-2.5-flash or local gemma).")
    args = parser.parse_args()

    # Make sure we have an API key if using Gemini or OpenAI
    if "gemini" in args.model.lower() and not os.environ.get("GEMINI_API_KEY"):
        logger.warning("GEMINI_API_KEY environment variable is missing. API calls will likely fail.")

    logger.info(f"Running Verifiers PoC Adapter test with model: {args.model}")
    
    # Create fake prompts and completions
    prompts = [
        f"[Context for ambiguity test {i}] Who is the president?" 
        for i in range(args.batch_size)
    ]
    completions = [
        f"Action: Clarify\nReasoning: I need more info.\nFacets: []\nResponse: Are you referring to the USA or another country? (Test {i})" 
        for i in range(args.batch_size)
    ]
    
    logger.info(f"Dispatching batch of {args.batch_size} asynchronous verifier tasks...")
    
    rewards = verifiers_reward_adapter(prompts, completions, judge_model=args.model)
    
    for i, reward in enumerate(rewards[:3]):
        logger.info(f"Sample {i} Reward: {reward}")
    
    if len(rewards) > 3:
        logger.info("...")
    
    logger.info("Test complete.")


if __name__ == "__main__":
    main()
