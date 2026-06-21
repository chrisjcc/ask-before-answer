import os

import streamlit as st
import weave
from dotenv import load_dotenv

from src.inference.pipeline import ClarificationPipeline

# Load environment variables for W&B authentication
load_dotenv()

# Initialize Weave to log all LLM traces to the project
weave.init("rl4aa/ask-before-answer")

# Set page config
st.set_page_config(
    page_title="AskBeforeAnswer: Clarification-Seeking LLM",
    page_icon="🤖",
    layout="wide",
)

# Initialize session state for model loading
if "pipeline" not in st.session_state:
    st.session_state.pipeline = None


def load_model():
    model_path = os.environ.get("MODEL_PATH", "Qwen/Qwen2.5-7B-Instruct")
    is_peft = os.environ.get("IS_PEFT", "false").lower() == "true"
    with st.spinner(f"Loading model from {model_path}..."):
        st.session_state.pipeline = ClarificationPipeline(model_path, is_peft=is_peft)


st.title("AskBeforeAnswer 🤖")
st.markdown(
    "This application demonstrates **clarification-seeking behavior** in Large "
    "Language Models. \n"
    "When given an ambiguous question, instead of hallucinating or assuming an "
    "intent, the model detects the ambiguity, explains the reasoning, identifies "
    "missing facets, and asks a targeted clarification question."
)

with st.sidebar:
    st.header("Settings")
    if st.button("Load Model"):
        load_model()

    st.markdown("### Example Queries")
    st.markdown("- When did The Simpsons first air?")
    st.markdown("- Who won the US Open?")
    st.markdown("- How do I make pasta?")

if st.session_state.pipeline is None:
    st.warning("Please load the model from the sidebar to begin.")
else:
    user_query = st.text_input("Enter a potentially ambiguous question:")

    if st.button("Generate Response") and user_query:
        with st.spinner("Analyzing..."):
            response = st.session_state.pipeline.generate(user_query)

            st.markdown("### Model Response")
            st.info(response)

            # Simple visualization of structured output
            if "Action: Clarify" in response:
                st.success(
                    "Model successfully detected ambiguity and requested clarification!"
                )
            elif "Action: Answer" in response:
                st.success(
                    "Model determined the question was unambiguous "
                    "and answered directly."
                )
