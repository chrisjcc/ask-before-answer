"""Multi-turn evaluation framework simulating Seeker and Provider agents.

This framework is inspired by ClarQ-LLM, enabling the evaluation of ClarifyOrAct 
models (Seeker Agents) in a dynamic, multi-turn loop with a simulated human 
(Provider Agent).
"""

import re
from typing import Any, Dict, List, Tuple

from google import genai

from ask_before_answer.inference.pipeline import ClarifyOrActPipeline


class ProviderAgent:
    """Simulates a human holding hidden disambiguation facets."""
    
    def __init__(self, original_question: str, disambiguations: List[dict]):
        self.original_question = original_question
        self.disambiguations = disambiguations
        self.client = genai.Client()
        self.system_prompt = (
            f"You are simulating a user who asked the following question: "
            f"'{self.original_question}'\n"
            f"However, this question is ambiguous. You secretly hold the following "
            f"valid interpretations and their answers: {disambiguations}\n\n"
            f"The AI (Seeker Agent) is asking you a clarifying question. "
            f"Pick ONE valid interpretation that best answers their clarifying "
            f"question, and respond succinctly as the user. Do not reveal other "
            f"interpretations they didn't ask for. Do not break character."
        )

    def reply(self, clarification_question: str) -> str:
        """Generate a reply based on the hidden facets."""
        prompt = (
            f"{self.system_prompt}\n\nSeeker asks: {clarification_question}\n"
            f"Your reply:"
        )
        response = self.client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        return response.text.strip()


class SeekerAgent:
    """Wraps the ClarifyOrAct pipeline to maintain state across multi-turn 
    interactions."""
    
    def __init__(self, pipeline: ClarifyOrActPipeline):
        self.pipeline = pipeline
        self.system_prompt = (
            "You are a helpful assistant. "
            "Given a question, you must decide whether it is ambiguous or not. "
            "Output MUST follow this format:\n"
            "Action: Clarify|Answer\n"
            "Reasoning: <your reasoning>\n"
            "Facets: <list of facets if ambiguous, else empty>\n"
            "Response: <clarifying question or direct answer>"
        )

    def step(self, history: List[dict]) -> str:
        """Take a step in the conversation given the dialogue history."""
        messages = [{"role": "system", "content": self.system_prompt}] + history
        return self.pipeline.generate_from_messages(messages)


def extract_action_and_response(raw_text: str) -> Tuple[str, str]:
    """Helper to parse the Clarify|Answer output from the Seeker agent."""
    action = "Answer"
    response = raw_text
    
    action_match = re.search(r"Action:\s*(Clarify|Answer)", raw_text, re.IGNORECASE)
    if action_match:
        action = action_match.group(1).capitalize()
        
    resp_match = re.search(r"Response:\s*(.*)", raw_text, re.IGNORECASE | re.DOTALL)
    if resp_match:
        response = resp_match.group(1).strip()
        
    return action, response


def simulate_conversation(
    seeker: SeekerAgent,
    provider: ProviderAgent,
    question: str,
    max_turns: int = 3
) -> Dict[str, Any]:
    """Simulate a multi-turn conversation between Seeker and Provider.
    
    Returns:
        dict: Contains simulation results including success boolean,
              total turns, complete dialogue history, and final response.
    """
    history = [{"role": "user", "content": question}]
    
    for turn in range(max_turns):
        # Seeker's turn
        raw_output = seeker.step(history)
        action, response = extract_action_and_response(raw_output)
        
        # Append seeker's output as an assistant message
        history.append({"role": "assistant", "content": response})
        
        if action == "Answer":
            return {
                "success": True,
                "turns": turn + 1,
                "history": history,
                "final_answer": response,
                "raw_output": raw_output
            }
            
        # Provider's turn
        provider_reply = provider.reply(response)
        history.append({"role": "user", "content": provider_reply})
        
    # Reached max turns without the Seeker deciding to act
    return {
        "success": False,
        "turns": max_turns,
        "history": history,
        "final_answer": None,
        "raw_output": None
    }
