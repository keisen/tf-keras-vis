from abc import ABC, abstractmethod

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K


class Regularizer(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, seed_inputs):
        raise NotImplementedError()


class TotalVariation(Regularizer):
    def __init__(self, weight=10.):
        self.weight = weight

    def __call__(self, seed_inputs):
        tv = 0.
        for x in seed_inputs:
            tv += (tf.image.total_variation(x) / np.prod(x.shape[1:]))
        return self.weight * tv


class L2Norm(Regularizer):
    def __init__(self, weight=10.):
        self.weight = weight

    def __call__(self, seed_inputs):
        norm = 0.
        for x in seed_inputs:
            norm += K.mean(K.l2_normalize(x))
        return self.weight * norm
