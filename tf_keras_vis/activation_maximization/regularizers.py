from abc import ABC, abstractmethod

import numpy as np
import tensorflow as tf


class Regularizer(ABC):
    """Abstract class for defining a regularizer.
    """
    def __init__(self, name) -> None:
        """
        Args:
            name: Instance name.
        """
        self.name = name

    @abstractmethod
    def __call__(self, input_value) -> tf.Tensor:
        """Implement regularization.

        Args:
            input_value: A tf.Tensor that indicates the value to input to the model.

        Returns:
            tf.Tensor: Regularization value with respect to the input value.

        Raises:
            NotImplementedError: This method must be overwritten.
        """
        raise NotImplementedError()


class TotalVariation2D(Regularizer):
    """A regularizer that introduces Total Variation.
    """
    def __init__(self, weight=10.0, name='TotalVariation2D') -> None:
        """
        Args:
            weight: This value will be apply to TotalVariation values.
                Defaults to 10.0.
            name : Instance name.
                Defaults to 'TotalVariation2D'.
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
    """
    def __init__(self, weight=10., p=2, name='Norm') -> None:
        """
        Args:
            weight: This weight will be apply to TotalVariation values.
                Defaults to 10.
            p: Order of the norm. Defaults to 2.
            name: Instance name. Defaults to 'Norm'. Defaults to 'Norm'.
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
