import warnings
from abc import ABC, abstractmethod

import numpy as np
import tensorflow as tf
from deprecated import deprecated

warnings.warn(('`tf_keras_vis.utils.regularizers` module is deprecated. '
               'Please use `tf_keras_vis.activation_maximization.regularizers` instead.'),
              DeprecationWarning)


@deprecated(version='0.7.0',
            reason="Please use `tf_keras_vis.activation_maximization.regularizers.Regularizer`"
            " class instead of this.")
class LegacyRegularizer(ABC):
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def __call__(self, inputs):
        raise NotImplementedError()


Regularizer = LegacyRegularizer


@deprecated(version='0.7.0',
            reason="The class has a bug that the calculated value is incorrect (too small) "
            "when the `batch_size` is greater than one. So please use "
            "`tf_keras_vis.activation_maximization.regularizers.TotalVariation2D`"
            " class instead of this.")
class TotalVariation2D(LegacyRegularizer):
    def __init__(self, weight=10., name='TotalVariation2D'):
        super().__init__(name)
        self.weight = weight

    def __call__(self, overall_inputs):
        tv = 0.
        for X in overall_inputs:
            tv += tf.image.total_variation(X) / np.prod(X.shape)
        return self.weight * tv


@deprecated(
    version='0.6.0',
    reason="Please use `tf_keras_vis.activation_maximization.regularizers.TotalVariation2D`"
    " class instead of this.")
class TotalVariation(TotalVariation2D):
    def __init__(self, weight=10.):
        super().__init__(weight=weight, name='TotalVariation')  # pragma: no cover


@deprecated(version='0.7.0',
            reason="The class has a bug that the calculated value is incorrect (too small). "
            "So please use `tf_keras_vis.activation_maximization.regularizers.Norm`"
            " class instead of this.")
class Norm(LegacyRegularizer):
    def __init__(self, weight=10., p=2, name='Norm'):
        super().__init__(name)
        self.weight = weight
        self.p = p

    def __call__(self, overall_inputs):
        norm = 0.
        for X in overall_inputs:
            X = tf.reshape(X, (X.shape[0], -1))
            norm += tf.norm(X, ord=self.p, axis=-1) / X.shape[1]
        return self.weight * norm


@deprecated(version='0.6.0',
            reason="Please use `tf_keras_vis.activation_maximization.regularizers.Norm`"
            " class instead of this.")
class L2Norm(Norm):
    def __init__(self, weight=10.):
        super().__init__(weight=weight, p=2, name='L2Norm')  # pragma: no cover
