try:
    import torch

    print("PyTorch loaded successfully")
except Exception as e:
    print(f"Failed to load torch: {e}")

try:
    import transformers.modeling_utils

    print("Transformers modeling_utils loaded successfully")
except Exception as e:
    print(f"REAL IMPORT ERROR: {e}")
    raise
