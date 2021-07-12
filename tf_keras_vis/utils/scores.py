from abc import ABC, abstractmethod
from typing import Union

import tensorflow as tf

from . import listify


class Score(ABC):
    """Abstract class for defining a score function.
    """
    def __init__(self, name=None) -> None:
        """
        Args:
            name: Instance name. Defaults to None.
        """
        self.name = name

    @abstractmethod
    def __call__(self, output) -> Union[tf.Tensor, list, tuple]:
        """Implement collecting scores that are used in visualization modules.

        Args:
            output: A tf.Tensor that indicates a model output value.

        Raises:
            NotImplementedError: This method must be overwritten.

        Returns:
            Score values.
        """
        raise NotImplementedError()


class InactiveScore(Score):
    """A score function that deactivate model output passed to `__call__()`.

    With a multiple output model, you can use this
    if you want a output to be excluded from targets of calculating gradients.
    """
    def __init__(self) -> None:
        super().__init__('InactiveScore')

    def __call__(self, output) -> tf.Tensor:
        return output * 0.0


class BinaryScore(Score):
    """A score function that collects the scores from model output
    which is for binary classification.
    """
    def __init__(self, target_values) -> None:
        """
        Args:
            target_values: A bool or a list of them.

        Raises:
            ValueError: When target_values is None or an empty list.
        """
        super().__init__('BinaryScore')
        self.target_values = listify(target_values, return_empty_list_if_none=False)
        if None in self.target_values:
            raise ValueError(f"Can't accept None value. target_values: {target_values}")
        self.target_values = [bool(v) for v in self.target_values]
        if len(self.target_values) == 0:
            raise ValueError(f"target_values is required. target_values: {target_values}")

    def __call__(self, output) -> tf.Tensor:
        if not (output.ndim == 2 and output.shape[1] == 1):
            raise ValueError(f"`output` shape must be (batch_size, 1), but was {output.shape}")
        output = tf.reshape(output, (-1, ))
        target_values = self.target_values
        if len(target_values) == 1 and len(target_values) < output.shape[0]:
            target_values = target_values * output.shape[0]
        return (2 * tf.constant(target_values, dtype=output.dtype) - 1.0) * output


class CategoricalScore(Score):
    """A score function that collects the scores from model output
    which is for categorical classification.
    """
    def __init__(self, indices) -> None:
        """
        Args:
            indices: An integer or a list of them.

        Raises:
            ValueError: When indices is None or an empty list.
        """
        super().__init__('CategoricalScore')
        self.indices = listify(indices, return_empty_list_if_none=False)
        if None in self.indices:
            raise ValueError(f"Can't accept None. indices: {indices}")
        if len(self.indices) == 0:
            raise ValueError(f"`indices` is required. indices: {indices}")

    def __call__(self, output) -> tf.Tensor:
        if output.ndim < 2:
            raise ValueError("`output` ndim must be 2 or more (batch_size, ..., channels), "
                             f"but was {output.ndim}")
        if output.shape[-1] <= max(self.indices):
            raise ValueError(
                f"Invalid index value. indices: {self.indices}, output.shape: {output.shape}")
        indices = self.indices
        if len(indices) == 1 and len(indices) < output.shape[0]:
            indices = indices * output.shape[0]
        score = [output[i, ..., index] for i, index in enumerate(indices)]
        score = tf.stack(score, axis=0)
        score = tf.math.reduce_mean(score, axis=tuple(range(score.ndim))[1:])
        return score
