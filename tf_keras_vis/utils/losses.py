import warnings

warnings.warn(('`tf_keras_vis.utils.losses` module is deprecated. '
               'Please use `tf_keras_vis.utils.scores` instead.'), DeprecationWarning)

from .scores import BinaryScore as BinaryLoss  # noqa: F401 E402
from .scores import CategoricalScore as CategoricalLoss  # noqa: F401 E402
from .scores import InactiveScore as InactiveLoss  # noqa: F401 E402
from .scores import Score as Loss  # noqa: F401 E402
