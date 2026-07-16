import sys

# Workaround for persistent PEFT / Transformers BloomPreTrainedModel
# import bug in CI environments
try:
    import transformers
    if not hasattr(transformers, "BloomPreTrainedModel"):
        class DummyBloomPreTrainedModel:
            pass

        transformers.BloomPreTrainedModel = (
            DummyBloomPreTrainedModel
        )
        sys.modules["transformers"].BloomPreTrainedModel = (
            DummyBloomPreTrainedModel
        )
except ImportError:
    pass
