from abc import ABC, abstractmethod

import numpy as np
import tensorflow as tf


class Regularizer(ABC):
    """Abstract class for defining a regularizer.

    Attributes:
        name (str): Instance name.
    """
    def __init__(self, name) -> None:
        """Constructor.

        Args:
            name (str): Instance name.
        """
        self.name = name

    @abstractmethod
    def __call__(self, input_value) -> tf.Tensor:
        """Implement regularization.

        Args:
            input_value (tf.Tensor): An value to input to the model.

        Returns:
            tf.Tensor: Regularization value with respect to the input value.

        Raises:
            NotImplementedError: This method must be overwritten.
        """
        raise NotImplementedError()


class TotalVariation2D(Regularizer):
    """A regularizer that introduces Total Variation.

    Attributes:
        weight (float): This weight will be apply to TotalVariation values.
        name (str): Instance name. Defaults to 'TotalVariation2D'.
    Todo:
        * Write examples
    """
    def __init__(self, weight=10.0, name='TotalVariation2D') -> None:
        """Constructor.

        Args:
            weight (float, optional): This value will be apply to TotalVariation values.
                Defaults to 10.0.
            name (str, optional): Instance name.. Defaults to 'TotalVariation2D'.
        """
        super().__init__(name)
        self.weight = float(weight)

    def __call__(self, input_value) -> tf.Tensor:
        if len(input_value.shape) != 4:
            raise ValueError("seed_input's shape must be (batch_size, height, width, channels), "
                             f"but was {input_value.shape}.")
        tv = tf.image.total_variation(input_value)
        tv /= np.prod(input_value.shape[1:], dtype=np.float)
        tv *= self.weight
        return tv


class Norm(Regularizer):
    """A regularizer that introduces Norm.

    Attributes:
        weight (float): This weight will be apply to TotalVariation values.
        p  (int): Order of the norm.
        name (str): Instance name. Defaults to 'Norm'.
    Todo:
        * Write examples
    """
    def __init__(self, weight=10., p=2, name='Norm') -> None:
        """Constructor.

        Args:
            weight (float, optional): This weight will be apply to TotalVariation values.
                Defaults to 10.
            p (int, optional): Order of the norm. Defaults to 2.
            name (str, optional): Instance name. Defaults to 'Norm'. Defaults to 'Norm'.
        """
        super().__init__(name)
        self.weight = float(weight)
        self.p = int(p)

    def __call__(self, input_value) -> tf.Tensor:
        input_value = tf.reshape(input_value, (input_value.shape[0], -1))
        norm = tf.norm(input_value, ord=self.p, axis=1)
        norm /= (float(input_value.shape[1])**(1.0 / float(self.p)))
        norm *= self.weight
        return norm
