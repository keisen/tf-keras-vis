from abc import ABC, abstractmethod

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

from tf_keras_vis.utils import listify


class Loss(ABC):
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def __call__(self, output):
        raise NotImplementedError()


class CategoricalScore(Loss):
    def __init__(self, indices, depth):
        super().__init__('CategoricalScore')
        self.indices = listify(indices)
        self.depth = depth

    def __call__(self, output):
        score = output * tf.one_hot(self.indices, self.depth)
        return K.mean(score, axis=tuple(range(len(score.shape))[1:]))


class SmoothedCategoricalScore(Loss):
    def __init__(self, indices, epsilon=0.05):
        super().__init__('CategoricalSmoothedScore')
        self.indices = listify(indices)
        self.epsilon = epsilon

    def __call__(self, output):
        smoothing_label = np.full(output.shape, self.epsilon / (np.prod(output.shape) - 1.))
        for i in self.indices:
            smoothing_label[..., i] = 1. - self.epsilon
        score = output * smoothing_label
        return K.mean(score, axis=tuple(range(len(score.shape))[1:]))
