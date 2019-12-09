from abc import ABC, abstractmethod

import numpy as np
import tensorflow as tf


class InputModifier(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, seed_input):
        raise NotImplementedError()


class Jitter(InputModifier):
    def __init__(self, jitter=0.05):
        super(Jitter, self).__init__()
        self.jitter = None
        self._jitter = jitter

    def __call__(self, seed_input):
        if self.jitter is None:
            shape = seed_input.shape[1:-1]
            self.axis = list(range(1, 1 + shape.rank))
            self.jitter = [
                dim * self._jitter if self._jitter < 1. else self._jitter for dim in shape
            ]
        return tf.roll(seed_input, [np.random.randint(-j, j + 1) for j in self.jitter], self.axis)
