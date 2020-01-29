from abc import ABC, abstractmethod

import numpy as np
from tensorflow.keras import backend as K

from tf_keras_vis.utils import listify


class Loss(ABC):
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def __call__(self, output):
        raise NotImplementedError()


class SmoothedLoss(Loss):
    def __init__(self, indices, epsilon=0.05):
        super().__init__('SmoothedLoss')
        self.indices = listify(indices)
        self.epsilon = epsilon

    def __call__(self, output):
        smoothing_label = np.full(output.shape, self.epsilon / (np.prod(output.shape) - 1.))
        for i in self.indices:
            smoothing_label[..., i] += 1. - (self.epsilon / len(self.indices))
        loss = output * smoothing_label
        return K.sum(loss) / len(self.indices)
