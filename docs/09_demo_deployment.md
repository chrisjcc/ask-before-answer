# Demo Application Deployment

The Hugging Face Demo is a separate Streamlit inference application.

Its source code consists of:
```text
app/
├── app.py
└── requirements.txt
```
and the minimal portion of the Python package required for inference (`src/ask_before_answer/inference/pipeline.py`). 
The application does **not** require the data, training, or evaluation modules, keeping the deployment payload extremely lightweight.

## Deployment Workflow

The HF Demo application is deployed through a GitHub Release rather than through the model publication command.

The workflow is defined in `.github/workflows/deploy-hf-demo.yml`.
It is triggered when a GitHub Release is published:

```yaml
on:
  release:
    types: [published]
```

The deployment sequence is:

```text
GitHub Release
      │
      ▼
deploy-hf-demo.yml (GitHub Action)
      │
      ▼
Check out released source revision
      │
      ▼
Construct minimal app payload
      │
      ▼
Push to Hugging Face Space repository
      │
      ▼
HF builds Docker image
      │
      ▼
Streamlit application goes live
```

The deployed application loads the production AskBeforeAnswer model from the Hugging Face model repository at runtime. This separation allows the model and application to evolve independently.
