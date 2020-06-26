from abc import ABC, abstractmethod

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K


class Regularizer(ABC):
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def __call__(self, inputs):
        raise NotImplementedError()


class TotalVariation(Regularizer):
    def __init__(self, weight=10.):
        super().__init__('TotalVariation')
        self.weight = weight

    def __call__(self, overall_inputs):
        tv = 0.
        for x in overall_inputs:
            tv += K.mean(tf.image.total_variation(x)) / np.prod(x.shape)
        return (self.weight * tv) / len(overall_inputs)


class L2Norm(Regularizer):
    def __init__(self, weight=10.):
        super().__init__('L2Norm')
        self.weight = weight

    def __call__(self, overall_inputs):
        norm = 0.
        for x in overall_inputs:
            norm += K.mean(K.l2_normalize(x))
        return (self.weight * norm) / len(overall_inputs)
