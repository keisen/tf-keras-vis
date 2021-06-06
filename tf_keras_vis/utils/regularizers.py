import warnings

warnings.warn(
    "`tf_keras_vis.utils.regularizers` module is deprecated. "
    "Please use `tf_keras_vis.activation_maximization.regularizers` instead.", DeprecationWarning)

from ..activation_maximization.regularizers import (  # noqa: E402,F401
    L2Norm, Norm, Regularizer, TotalVariation, TotalVariation2D)
