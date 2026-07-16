try:
    import torch  # noqa: F401

    print("PyTorch loaded successfully")
except Exception as e:
    print(f"Failed to load torch: {e}")

try:
    import transformers.modeling_utils  # noqa: F401

    print("Transformers modeling_utils loaded successfully")
except Exception as e:
    print(f"REAL IMPORT ERROR: {e}")
    raise
