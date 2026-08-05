FROM python:3.10-slim

LABEL org.opencontainers.image.source="https://github.com/chrisjcc/ask-before-answer"
LABEL org.opencontainers.image.description="AskBeforeAnswer: An RLHF-trained agent that clarifies ambiguity."
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY app/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# PATCH PEFT: HuggingFace hasn't fixed this bug yet. We manually remove the broken
# BloomPreTrainedModel import directly from the installed PEFT source code.
RUN sed -i 's/from transformers import BloomPreTrainedModel/BloomPreTrainedModel = type("BloomPreTrainedModel", (object,), {})/g' /usr/local/lib/python3.10/site-packages/peft/utils/constants.py || true

# Copy source code
COPY . .

# Install the package itself
RUN pip install -e .

EXPOSE 8501



CMD ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0", "--browser.gatherUsageStats=false"]
