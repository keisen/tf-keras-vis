import warnings

warnings.warn(('`tf_keras_vis.utils.callbacks` module is deprecated. '
               'Please use `tf_keras_vis.activation_maximization.callbacks` instead.'),
              DeprecationWarning)

from ..activation_maximization.callbacks import Callback as OptimizerCallback  # noqa: F401 E402
from ..activation_maximization.callbacks import GifGenerator2D  # noqa: F401 E402
from ..activation_maximization.callbacks import GifGenerator2D as GifGenerator  # noqa: F401 E402
from ..activation_maximization.callbacks import PrintLogger as Print  # noqa: F401 E402
