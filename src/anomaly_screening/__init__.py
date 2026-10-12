"""Frozen Anomaly Screening implementation; experimental trainable policies are archived."""

__all__ = [
    "BaselineDecisionPolicy",
    "BaselineFusionPolicy",
    "BaselineSimilarityPolicy",
    "ScreeningState",
    "DecisionOutput",
    "DecisionPolicy",
    "FusionOutput",
    "FrozenScreening",
    "ModularPipelineOutput",
    "MultiScaleFusionPolicy",
    "SimilarityOutput",
    "SimilarityPolicy",
]


def __getattr__(name):
    # Cache readers must not load the encoder/PyTorch runtime.
    if name in __all__:
        from . import frozen

        return getattr(frozen, name)
    raise AttributeError(name)
