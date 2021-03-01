from abc import ABC, abstractmethod

import numpy as np
import tensorflow as tf
from deprecated import deprecated
from packaging.version import parse as version

if version(tf.version.VERSION) < version("2.4.0"):
    from tensorflow.keras.mixed_precision.experimental import global_policy
else:
    from tensorflow.keras.mixed_precision import global_policy


class Regularizer(ABC):
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def __call__(self, inputs):
        raise NotImplementedError()


class TotalVariation2D(Regularizer):
    def __init__(self, weight=10., name='TotalVariation2D'):
        super().__init__(name)
        self.weight = weight
        self.policy = global_policy()

    def __call__(self, overall_inputs):
        tv = 0.
        for X in overall_inputs:
            if self.policy.variable_dtype != self.policy.compute_dtype:
                X = tf.cast(X, dtype=self.policy.variable_dtype)
            tv += tf.image.total_variation(X) / np.prod(X.shape)
            if self.policy.variable_dtype != self.policy.compute_dtype:
                tv = tf.cast(tv, dtype=self.policy.compute_dtype)
        return self.weight * tv


@deprecated(version='0.6.0',
            reason="Please use TotalVariation2D class instead of TotalVariation class.")
class TotalVariation(TotalVariation2D):
    def __init__(self, weight=10.):
        super().__init__(weight=weight, name='TotalVariation')


class Norm(Regularizer):
    def __init__(self, weight=10., p=2, name='Norm'):
        super().__init__(name)
        self.weight = weight
        self.p = p
        self.policy = global_policy()

    def __call__(self, overall_inputs):
        norm = 0.
        for X in overall_inputs:
            if self.policy.variable_dtype != self.policy.compute_dtype:
                X = tf.cast(X, dtype=self.policy.variable_dtype)
            X = tf.reshape(X, (X.shape[0], -1))
            norm += tf.norm(X, ord=self.p, axis=-1) / X.shape[-1]
            if self.policy.variable_dtype != self.policy.compute_dtype:
                norm = tf.cast(norm, dtype=self.policy.compute_dtype)
        return self.weight * norm


@deprecated(version='0.6.0', reason="Please use Norm class instead of L2Norm class.")
class L2Norm(Norm):
    def __init__(self, weight=10.):
        super().__init__(weight=weight, p=2, name='L2Norm')
