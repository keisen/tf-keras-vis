from abc import ABC, abstractmethod

import numpy as np
import tensorflow as tf
from scipy.ndimage import rotate


class InputModifier(ABC):
    """Abstract class for defining an input modifier.
    """
    @abstractmethod
    def __call__(self, seed_input):
        """Implement modification to the input before processing gradient descent.

        # Arguments:
            seed_input: A tf.Tensor.
        # Returns:
            The modified `seed_input`.
        """
        raise NotImplementedError()


class Jitter(InputModifier):
    def __init__(self, jitter=8):
        """Implements an input modifier that introduces random jitter.
            Jitter has been shown to produce crisper activation maximization images.

        # Arguments:
            jitter: Integer. The amount of jitter to apply.
        """
        self.jitter = int(jitter)

    def __call__(self, seed_input):
        ndim = len(seed_input.shape)
        seed_input = tf.roll(seed_input,
                             shift=tuple(np.random.randint(-self.jitter, self.jitter, ndim - 2)),
                             axis=tuple(range(ndim)[1:-1]))
        return seed_input


class Rotate(InputModifier):
    def __init__(self, degree=3.):
        """Implements an input modifier that introduces random rotation.
            Rotate has been shown to produce crisper activation maximization images.

        # Arguments:
            degree: Integer or float. The amount of rotation to apply.
        """
        self.rg = float(degree)

    def __call__(self, seed_input):
        if tf.is_tensor(seed_input):
            seed_input = seed_input.numpy()
        if seed_input.dtype == np.float16:
            seed_input = seed_input.astype(np.float32)
        seed_input = rotate(seed_input,
                            np.random.uniform(-self.rg, self.rg),
                            axes=tuple(range(len(seed_input.shape))[1:-1]),
                            reshape=False,
                            mode='nearest',
                            order=1,
                            prefilter=True)
        return tf.constant(seed_input)
