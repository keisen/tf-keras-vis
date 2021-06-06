import warnings

warnings.warn(
    "`tf_keras_vis.utils.input_modifiers` module is deprecated. "
    "Please use `tf_keras_vis.activation_maximization.input_modifiers` instead.",
    DeprecationWarning)

from ..activation_maximization.input_modifiers import (  # noqa: E402,F401
    InputModifier, Jitter, Rotate, Rotate2D)
