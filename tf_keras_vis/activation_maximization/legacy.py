import tensorflow as tf

from . import ActivationMaximization as ActivationMaximizationBase
from ..utils.input_modifiers import Jitter, Rotate2D
from ..utils.regularizers import Norm, TotalVariation2D


class ActivationMaximization(ActivationMaximizationBase):
    """ActivationMaximization of legacy version.

        This class was introduced in v0.7.0 to keep compatibility with the past versions.
    """
    def __call__(self,
                 score,
                 seed_input=None,
                 input_range=(0, 255),
                 input_modifiers=[Jitter(jitter=8), Rotate2D(degree=3)],
                 regularizers=[TotalVariation2D(weight=1.),
                               Norm(weight=1., p=2)],
                 steps=200,
                 optimizer=None,
                 normalize_gradient=None,
                 gradient_modifier=None,
                 callbacks=None,
                 training=False,
                 unconnected_gradients=tf.UnconnectedGradients.NONE):
        arguments = locals().items()
        arguments = ((k, v) for k, v in arguments if k != 'self')
        arguments = ((k, v) for k, v in arguments if not k.startswith('_'))
        arguments = dict(arguments)
        if optimizer is None:
            arguments['optimizer'] = tf.optimizers.RMSprop(1.0, 0.95)
        return super().__call__(_compatible_mode=True, **arguments)
