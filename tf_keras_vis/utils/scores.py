from abc import ABC, abstractmethod

import tensorflow as tf

from tf_keras_vis.utils import listify


class Score(ABC):
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def __call__(self, output):
        raise NotImplementedError()


class InactiveScore(Score):
    def __init__(self):
        super().__init__('InactiveScore')

    def __call__(self, output):
        return output * 0.0


class BinaryScore(Score):
    def __init__(self, target_values):
        '''
        target_values: bool values.  When the type of values is not bool,
            they will be casted to bool.
        '''
        super().__init__('BinaryScore')
        self.target_values = listify(target_values, return_empty_list_if_none=False)
        if None in self.target_values:
            raise ValueError("Can't accept None value. [{}]".format(target_values))
        self.target_values = [bool(v) for v in self.target_values]
        if len(self.target_values) == 0:
            raise ValueError('target_values is required. [{}]'.format(target_values))

    def __call__(self, output):
        if output.ndim != 1 and not (output.ndim == 2 and output.shape[1] == 1):
            raise ValueError("output shape must be (batch_size, 1), but was {}".format(
                output.shape))
        output = tf.reshape(output, (-1, ))
        target_values = self.target_values
        if len(target_values) == 1 and len(target_values) < output.shape[0]:
            target_values = target_values * output.shape[0]
        score = [val if positive else 1.0 - val for val, positive in zip(output, target_values)]
        return score


class CategoricalScore(Score):
    def __init__(self, indices):
        super().__init__('CategoricalScore')
        self.indices = listify(indices, return_empty_list_if_none=False)
        if None in self.indices:
            raise ValueError("Can't accept None. indices: [{}]".format(indices))
        if len(self.indices) == 0:
            raise ValueError('indices is required. [{}]'.format(indices))

    def __call__(self, output):
        if output.ndim < 2:
            raise ValueError("output ndim must be 2 or more (batch_size, ..., channels), "
                             "but was {}".format(output.ndim))
        if output.shape[-1] <= max(self.indices):
            raise ValueError("Invalid index value. indices: {}, output.shape: {}".format(
                self.indices, output.shape))
        indices = self.indices
        if len(indices) == 1 and len(indices) < output.shape[0]:
            indices = indices * output.shape[0]
        score = [output[i, ..., index] for i, index in enumerate(indices)]
        score = tf.stack(score, axis=0)
        score = tf.math.reduce_mean(score, axis=tuple(range(score.ndim))[1:])
        return score
