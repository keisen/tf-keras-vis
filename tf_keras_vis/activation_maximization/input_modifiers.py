from abc import ABC, abstractmethod

import numpy as np
import tensorflow as tf
from deprecated import deprecated
from scipy.ndimage import rotate


class InputModifier(ABC):
    """Abstract class for defining an input modifier.
    """
    @abstractmethod
    def __call__(self, seed_input):
        """Implement modification to the input before processing gradient descent.

        Args:
            seed_input (tf.Tensor): A value to input to model.

        Raises:
            NotImplementedError: This method must be overwritten.
        """
        raise NotImplementedError()


class Jitter(InputModifier):
    """An input modifier that introduces random jitter.
        Jitter has been shown to produce crisper activation maximization images.

    Attributes:
        jitter (int): The amount of jitter to apply.
    Todo:
        * Write examples
    """
    def __init__(self, jitter=8):
        """Constructor.

        Args:
            jitter (int, optional): The amount of jitter to apply. Defaults to 8.
        """
        self.jitter = int(jitter)

    def __call__(self, seed_input):
        ndim = len(seed_input.shape)
        if ndim < 3:
            raise ValueError("The dimensions of seed_input must be 3 or more "
                             f"(batch_size, ..., channels), but was {ndim}")
        seed_input = tf.roll(seed_input,
                             shift=tuple(np.random.randint(-self.jitter, self.jitter, ndim - 2)),
                             axis=tuple(range(ndim)[1:-1]))
        return seed_input


class Rotate2D(InputModifier):
    """An input modifier that introduces random rotation.

    Args:
        degree (float): The amount of rotation to apply.
    Todo:
        * Write examples
    """
    def __init__(self, degree=3.0):
        """Constructor.

        Args:
            degree (int|float, optional): The amount of rotation to apply. Defaults to 3.0.
        """
        self.degree = float(degree)

    def __call__(self, seed_input):
        ndim = len(seed_input.shape)
        if ndim != 4:
            raise ValueError("seed_input shape must be (batch_size, height, width, channels),"
                             f" but was {seed_input.shape}")
        if tf.is_tensor(seed_input):
            seed_input = seed_input.numpy()
        seed_input = rotate(seed_input,
                            np.random.uniform(-self.degree, self.degree),
                            axes=tuple(range(len(seed_input.shape))[1:-1]),
                            reshape=False,
                            mode='nearest',
                            order=1,
                            prefilter=True)
        seed_input = tf.constant(seed_input)
        return seed_input


@deprecated(version='0.6.2', reason="Please use Rotate2D class instead of Rotate class.")
class Rotate(Rotate2D):
    def __init__(self, degree=3.0):
        super().__init__(degree=3.0)  # pragma: no cover
