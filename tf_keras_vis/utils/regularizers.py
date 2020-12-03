from abc import ABC, abstractmethod

import numpy as np
import tensorflow as tf

from deprecated import deprecated


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

    def __call__(self, overall_inputs):
        tv = 0.
        for X in overall_inputs:
            tv += tf.image.total_variation(X) / np.prod(X.shape)
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

    def __call__(self, overall_inputs):
        norm = 0.
        for X in overall_inputs:
            X = tf.reshape(X, (X.shape[0], -1))
            norm += tf.norm(X, ord=self.p, axis=-1) / X.shape[-1]
        return self.weight * norm


@deprecated(version='0.6.0', reason="Please use Norm class instead of L2Norm class.")
class L2Norm(Norm):
    def __init__(self, weight=10.):
        super().__init__(weight=weight, lp=2, name='L2Norm')
