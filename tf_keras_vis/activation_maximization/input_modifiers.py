from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import tensorflow as tf
from scipy.ndimage.interpolation import rotate, zoom


class InputModifier(ABC):
    """Abstract class for defining an input modifier.
    """
    @abstractmethod
    def __call__(self, seed_input) -> Union[np.ndarray, tf.Tensor]:
        """Implement modification to the input before processing gradient descent.

        Args:
            seed_input: An :obj:`numpy.ndarray` or a tf.Tensor that indicates a value to input to
                model.

        Returns:
            An :obj:`numpy.ndarray` or a tf.Tensor.

        Raises:
            NotImplementedError: This method must be overwritten.
        """
        raise NotImplementedError()


class Jitter(InputModifier):
    """An input modifier that introduces random jitter.
        Jitter has been shown to produce crisper activation maximization images.
    """
    def __init__(self, jitter=8) -> None:
        """
        Args:
            jitter: The amount of jitter to apply. Defaults to 8.
        """
        self.jitter = int(jitter)

    def __call__(self, seed_input) -> np.ndarray:
        ndim = len(seed_input.shape)
        if ndim < 3:
            raise ValueError("The dimensions of seed_input must be 3 or more "
                             f"(batch_size, ..., channels), but was {ndim}.")
        seed_input = tf.roll(seed_input,
                             shift=tuple(np.random.randint(-self.jitter, self.jitter, ndim - 2)),
                             axis=tuple(range(ndim)[1:-1]))
        return seed_input


class Rotate(InputModifier):
    """An input modifier that introduces random rotation.
    """
    def __init__(self, axes=(1, 2), degree=3.0) -> None:
        """
        Args:
            axes: The two axes that define the plane of rotation.
                Defaults to (1, 2).
            degree: The amount of rotation to apply. Defaults to 3.0.

        Raises:
            ValueError: When axes is not a tuple of two ints.
        """
        if type(axes) not in [list, tuple] or len(axes) != 2:
            raise ValueError(f"`axes` must be a tuple of two int values, but it was {axes}.")
        if not isinstance(axes[0], int) or not isinstance(axes[1], int):
            raise TypeError(f"`axes` must be consist of ints, but it was {axes}.")
        self.axes = axes
        self.degree = float(degree)
        self.random_generator = np.random.default_rng()

    def __call__(self, seed_input) -> np.ndarray:
        ndim = len(seed_input.shape)
        if ndim < 4:
            raise ValueError("The dimensions of seed_input must be 4 or more "
                             f"(batch_size, ..., channels), but was {ndim}.")
        if tf.is_tensor(seed_input):
            seed_input = seed_input.numpy()
        seed_input = rotate(seed_input,
                            self.random_generator.uniform(-self.degree, self.degree),
                            axes=self.axes,
                            reshape=False,
                            order=1,
                            mode='reflect',
                            prefilter=False)
        return seed_input


class Rotate2D(Rotate):
    """An input modifier for 2D that introduces random rotation.
    """
    def __init__(self, degree=3.0) -> None:
        """
        Args:
            degree: The amount of rotation to apply. Defaults to 3.0.
        """
        super().__init__(axes=(1, 2), degree=degree)


class Scale(InputModifier):
    """An input modifier that introduces randam scaling.
    """
    def __init__(self, low=0.9, high=1.1) -> None:
        """
        Args:
            low (float, optional): Lower boundary of the zoom factor. Defaults to 0.9.
            high (float, optional): Higher boundary of the zoom factor. Defaults to 1.1.
        """
        self.low = low
        self.high = high
        self.random_generator = np.random.default_rng()

    def __call__(self, seed_input) -> np.ndarray:
        ndim = len(seed_input.shape)
        if ndim < 3:
            raise ValueError("The dimensions of seed_input must be 3 or more "
                             f"(batch_size, ..., channels), but was {ndim}.")
        if tf.is_tensor(seed_input):
            seed_input = seed_input.numpy()
        shape = seed_input.shape
        _factor = factor = self.random_generator.uniform(self.low, self.high)
        factor *= np.ones(ndim - 2)
        factor = (1, ) + tuple(factor) + (1, )
        seed_input = zoom(seed_input, factor, order=1, mode='reflect', prefilter=False)
        if _factor > 1.0:
            indices = (self._central_crop_range(x, e) for x, e in zip(seed_input.shape, shape))
            indices = (slice(start, stop) for start, stop in indices)
            seed_input = seed_input[tuple(indices)]
        if _factor < 1.0:
            pad_width = [self._pad_width(x, e) for x, e in zip(seed_input.shape, shape)]
            seed_input = np.pad(seed_input, pad_width, 'mean')
        return seed_input

    def _central_crop_range(self, x, e):
        start = (x - e) // 2
        stop = start + e
        return start, stop

    def _pad_width(self, x, e):
        diff = e - x
        before = diff // 2
        after = diff - before
        return before, after
