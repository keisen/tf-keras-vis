import warnings
from abc import ABC
from contextlib import contextmanager
from inspect import signature

import imageio
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont

from ..utils import listify


class Callback(ABC):
    """Abstract class for defining callbacks.
    """
    def on_begin(self, **kwargs) -> None:
        """Called at the begin of optimization process.

        Args:
            kwargs (dict): The parameters that is passed to
                `tf_keras_vis.activation_maximization.ActivationMaximization#__call__()`.
        """
        pass

    def __call__(self, i, values, grads, score_values, outputs, regularizer_values, overall_score):
        """his function will be called within
            `tf_keras_vis.activation_maximization.ActivationMaximization#__call__()`.

        Args:
            i (int): The current number of optimizer iteration.
            values (list): The current `values`.
            grads (list): The gradient of input images with respect to `values`.
            score_values (list): A list of score values with respect to each the model outputs.
            outputs (list): A list of the model outputs.
            regularizer_values (list): A list of regularizer values.
            overall_score (list): A list of overall scores that includes
                score values and regularizer values.
        """
        pass

    def on_end(self):
        """Called at the end of optimization process.
        """
        pass


class PrintLogger(Callback):
    """Callback to print values during optimization.

    Attributes:
        interval (int): An integer that appears the interval of printing.
    Todo:
        * Write examples
    """
    def __init__(self, interval=10):
        """Constructor.

        Args:
            interval (int, optional): An integer that appears the interval of printing.
                Defaults to 10.
        """
        self.interval = interval

    def __call__(self, i, values, grads, score_values, outputs, regularizer_values, overall_score):
        i += 1
        if (i % self.interval == 0):
            tf.print('Steps: {:03d}\tScores: {},\tRegularization: {}'.format(
                i, self._tolist(score_values), self._tolist(regularizer_values)))

    def _tolist(self, ary):
        if isinstance(ary, list) or isinstance(ary, (np.ndarray, np.generic)):
            return [self._tolist(e) for e in ary]
        elif isinstance(ary, tuple):
            return tuple(self._tolist(e) for e in ary)
        elif tf.is_tensor(ary):
            return ary.numpy().tolist()
        else:
            return ary


class GifGenerator2D(Callback):
    """Callback to construct gif of optimized image.

    Attributes:
        path (str): The file path to save gif.
    Todo:
        * Write examples
    """
    def __init__(self, path):
        """Constructor.

        Args:
            path (str): The file path to save gif.
        """
        self.path = path

    def on_begin(self, **kwargs):
        self.data = None

    def __call__(self, i, values, grads, score_values, outputs, regularizer_values, overall_score):
        if self.data is None:
            self.data = [[] for i in range(len(values))]
        for n, value in enumerate(values):
            img = Image.fromarray(value[0].astype(np.uint8))  # 1st image in a batch
            ImageDraw.Draw(img).text((10, 10),
                                     "Step {}".format(i + 1),
                                     font=ImageFont.load_default())
            self.data[n].append(np.asarray(img))

    def on_end(self):
        path = self.path if self.path.endswith('.gif') else '{}.gif'.format(self.path)
        for i in range(len(self.data)):
            writer = None
            try:
                writer = imageio.get_writer(path, mode='I', loop=0)
                for data in self.data[i]:
                    writer.append_data(data)
            finally:
                if writer is not None:
                    writer.close()


@contextmanager
def managed_callbacks(callbacks=None, **kwargs):
    activated_callbacks = []
    try:
        for c in listify(callbacks):
            if len(signature(c.on_begin).parameters) == 0:
                warnings.warn("`Callback#on_begin()` now must accept keyword arguments.",
                              DeprecationWarning)
                c.on_begin()
            else:
                c.on_begin(**kwargs)
            activated_callbacks.append(c)
        yield activated_callbacks
        for _ in range(len(activated_callbacks)):
            activated_callbacks.pop(0).on_end()
    finally:
        for c in activated_callbacks:
            try:
                c.on_end()
            except Exception as e:
                tf.print("Exception args: ", e.args)
                pass
