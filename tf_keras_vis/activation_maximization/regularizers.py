from abc import ABC, abstractmethod

import numpy as np
import tensorflow as tf
from deprecated import deprecated


class Regularizer(ABC):
    """Abstract class for defining a regularizer.

    Attributes:
        name (str): Instance name.
    """
    def __init__(self, name):
        """Constructor.

        Args:
            name (str): Instance name. This will be include a log that is printed by
            `tf_keras_vis.activation_maximization.callbacks.PrintLogger`.
        """
        self.name = name

    @abstractmethod
    def __call__(self, inputs):
        """Implement regularization.

        Args:
            inputs (list): A list of tf.Tensor or tf.Variable.

        Raises:
            NotImplementedError: This method must be overwritten.
        """
        raise NotImplementedError()


class TotalVariation2D(Regularizer):
    """A regularizer that introduces Total Variation.

    Attributes:
        name (str): Instance name. Defaults to 'TotalVariation2D'.
        weight (float): This weight will be apply to TotalVariation values.
    Todo:
        * Write examples
    """
    def __init__(self, weight=10.0, name='TotalVariation2D'):
        """Constructor.

        Args:
            weight (float, optional): This value will be apply to TotalVariation values.
                Defaults to 10.0.
            name (str, optional): Instance name.. Defaults to 'TotalVariation2D'.
        """
        super().__init__(name)
        self.weight = weight

    def __call__(self, overall_inputs):
        tv = 0.
        for X in overall_inputs:
            tv += tf.image.total_variation(X) / np.prod(X.shape[1:])
        return self.weight * tv


@deprecated(version='0.6.0',
            reason="Please use TotalVariation2D class instead of this. "
            "This class can NOT support N-dim tensor, only supports 2-dim input.")
class TotalVariation(TotalVariation2D):
    def __init__(self, weight=10.0):
        super().__init__(weight=weight, name='TotalVariation')  # pragma: no cover


class Norm(Regularizer):
    """A regularizer that introduces Norm.

    Attributes:
        name (str): Instance name. Defaults to 'Norm'.
        weight (float): This weight will be apply to TotalVariation values.
        p  (int): Order of the norm.
    Todo:
        * Write examples
    """
    def __init__(self, weight=10., p=2, name='Norm'):
        """Constructor.

        Args:
            weight (float, optional): This weight will be apply to TotalVariation values.
                Defaults to 10.
            p (int, optional): Order of the norm. Defaults to 2.
            name (str, optional): Instance name. Defaults to 'Norm'. Defaults to 'Norm'.
        """
        super().__init__(name)
        self.weight = weight
        self.p = p

    def __call__(self, overall_inputs):
        norm = 0.
        for X in overall_inputs:
            X = tf.reshape(X, (X.shape[0], -1))
            norm += tf.norm(X, ord=self.p, axis=-1) / (X.shape[-1]**(1 / self.p))
        return self.weight * norm


@deprecated(version='0.6.0', reason="Please use Norm class instead of this.")
class L2Norm(Norm):
    def __init__(self, weight=10.):
        super().__init__(weight=weight, p=2, name='L2Norm')  # pragma: no cover
