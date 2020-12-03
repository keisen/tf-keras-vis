import warnings
warnings.warn(('`tf_keras_vis.utils.losses` module is deprecated. '
               'Please use `tf_keras_vis.utils.scores` instead.'), DeprecationWarning)

from tf_keras_vis.utils.scores import BinaryScore as BinaryLoss  # noqa: F401 E402
from tf_keras_vis.utils.scores import CategoricalScore as CategoricalLoss  # noqa: F401 E402
from tf_keras_vis.utils.scores import InactiveScore as InactiveLoss  # noqa: F401 E402
from tf_keras_vis.utils.scores import Score as Loss  # noqa: F401 E402
from tf_keras_vis.utils.scores import SmoothedCategoricalScore as SmoothedCategoricalLoss  # noqa: F401 E402 E501
