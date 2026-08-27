# ------------------------------------------------------------
# AskBeforeAnswer Hugging Face Space
# CPU-only inference and Streamlit demo
# ------------------------------------------------------------

FROM python:3.10-slim

LABEL org.opencontainers.image.source="https://github.com/chrisjcc/ask-before-answer"
LABEL org.opencontainers.image.description="AskBeforeAnswer: an RLHF-trained agent that clarifies ambiguity."

WORKDIR /app

# ------------------------------------------------------------
# System dependencies
# ------------------------------------------------------------

RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------
# Python dependencies
# ------------------------------------------------------------

COPY app/requirements.txt ./requirements.txt

RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ------------------------------------------------------------
# Application source
# ------------------------------------------------------------

COPY app ./app
COPY src ./src
COPY pyproject.toml ./

# ------------------------------------------------------------
# Install AskBeforeAnswer package
# ------------------------------------------------------------

RUN pip install --no-cache-dir -e .

# ------------------------------------------------------------
# Runtime configuration
# ------------------------------------------------------------

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TOKENIZERS_PARALLELISM=false

EXPOSE 8501

# ------------------------------------------------------------
# Start Streamlit
# ------------------------------------------------------------

CMD ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0", "--browser.gatherUsageStats=false"]
