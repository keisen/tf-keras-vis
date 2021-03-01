from abc import ABC, abstractmethod

import tensorflow as tf
import tensorflow.keras.backend as K

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
    '''
    target_values: A bool or int value [0, 1].
    '''
    def __init__(self, target_values):
        super().__init__('BinaryScore')
        self.target_values = [bool(v) for v in listify(target_values)]

    def __call__(self, output):
        output = tf.reshape(output, (output.shape[0], -1))
        score = [
            output[i] * (1.0 if positive else -1.0) for i, positive in range(self.target_values)
        ]
        score = tf.concat(score, axis=0)
        score = K.mean(score, axis=1)
        return score


class CategoricalScore(Score):
    def __init__(self, indices):
        super().__init__('CategoricalScore')
        self.indices = listify(indices)

    def __call__(self, output):
        output = tf.reshape(output, (output.shape[0], -1, output.shape[-1]))
        score = [output[i:i + 1, :, index] for i, index in enumerate(self.indices)]
        score = tf.concat(score, axis=0)
        score = K.mean(score, axis=1)
        return score
