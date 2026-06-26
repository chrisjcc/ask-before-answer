import glob
import os

import streamlit as st
import weave
import yaml
from dotenv import load_dotenv

from src.inference.pipeline import ClarificationPipeline

# Load environment variables for W&B authentication
load_dotenv()

# Initialize Weave to log all LLM traces to the project
if os.environ.get("WANDB_API_KEY"):
    try:
        wandb_entity = os.environ.get("WANDB_ENTITY", "rl4aa")
        wandb_project = os.environ.get("WANDB_PROJECT", "ask-before-answer")
        weave.init(f"{wandb_entity}/{wandb_project}")
    except Exception as e:
        print(f"Failed to initialize Weave: {e}")
else:
    print("WANDB_API_KEY not found. Weave tracing disabled.")

# Set page config
st.set_page_config(
    page_title="AskBeforeAnswer: Clarification-Seeking LLM",
    page_icon="🤖",
    layout="wide",
)

# Initialize session state for model loading
if "pipeline" not in st.session_state:
    st.session_state.pipeline = None


def get_available_models():
    models = {
        "AskBeforeAnswer (SFT+DPO Qwen2.5-7B)": {
            "path": "models/dpo/final",
            "is_peft": True,
        }
    }

    # Dynamically scan model configs for baselines
    for config_file in glob.glob("configs/model/*.yaml"):
        try:
            with open(config_file, "r") as f:
                cfg = yaml.safe_load(f)
                if cfg and "name" in cfg:
                    models[f"Base: {cfg['name']}"] = {
                        "path": cfg["name"],
                        "is_peft": False,
                    }
        except Exception as e:
            print(f"Skipping config {config_file}: {e}")

    return models


@st.cache_resource(show_spinner=False)
def load_pipeline(model_path: str, is_peft: bool):
    return ClarificationPipeline(model_path, is_peft=is_peft)


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
    available_models = get_available_models()
    selected_model_name = st.selectbox("Select Model", list(available_models.keys()))

    if st.button("Load Model"):
        import gc

        import torch

        # Aggressively clean up old model memory before loading the new one
        if st.session_state.pipeline is not None:
            del st.session_state.pipeline
            st.session_state.pipeline = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        model_info = available_models[selected_model_name]
        with st.spinner(f"Loading {selected_model_name}..."):
            st.session_state.pipeline = load_pipeline(
                model_info["path"], model_info["is_peft"]
            )

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
