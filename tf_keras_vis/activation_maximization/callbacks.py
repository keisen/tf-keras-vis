import warnings
from abc import ABC
from contextlib import contextmanager
from inspect import signature

import imageio
import numpy as np
import tensorflow as tf
from deprecated import deprecated
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

    def __call__(self, i, values, grads, scores, model_outputs, **kwargs) -> None:
        """This function will be called within
            `tf_keras_vis.activation_maximization.ActivationMaximization#__call__()`
            after gradient descent and updating input values.

        Args:
            i (int): The current number of optimizer iteration.
            values (list[tf.Tensor]): The current `values` that is clopped and modified.
            grads (list[tf.Tensor]): The gradients with respect to `values`.
            scores (list[tf.Tensor]): Score values with respect to each the model outputs.
            model_outputs (list[tf.Tensor]): A list of the model outputs.
            regularizations (list[Tuple[str,tf.Tensor]], optional): A list of regularizer values.
            overall_score (list[tf.Tensor], optional):
                Overall scores that includes the scores and regularization values.
        """
        pass

    def on_end(self) -> None:
        """Called at the end of optimization process.
        """
        pass


@deprecated(version='0.7.0', reason="Use `Progress` instead.")
class PrintLogger(Callback):
    """Callback to print values during optimization.

    Attributes:
        interval (int): An integer that appears the interval of printing.
    """
    def __init__(self, interval=10):
        """Constructor.

        Args:
            interval (int, optional): An integer that appears the interval of printing.
                Defaults to 10.
        """
        self.interval = interval

    def __call__(self, i, values, grads, scores, model_outputs, regularizations, **kwargs):
        i += 1
        if (i % self.interval == 0):
            tf.print('Steps: {:03d}\tScores: {},\tRegularization: {}'.format(
                i, self._tolist(scores), self._tolist(regularizations)))

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
    """Callback to construct a gif of optimized image.

    Attributes:
        path (str): The file path to save gif.
    Todo:
        * Write examples
    """
    def __init__(self, path) -> None:
        """Constructor.

        Args:
            path (str): The file path to save gif.
        """
        self.path = path

    def on_begin(self, **kwargs) -> None:
        self.data = None

    def __call__(self, i, values, *args, **kwargs) -> None:
        if self.data is None:
            self.data = [[] for _ in range(len(values))]
        for n, value in enumerate(values):
            img = Image.fromarray(value[0].astype(np.uint8))  # 1st image in the batch
            ImageDraw.Draw(img).text((10, 10), f"Step {i + 1}", font=ImageFont.load_default())
            self.data[n].append(np.asarray(img))

    def on_end(self) -> None:
        path = self.path if self.path.endswith(".gif") else f"{self.path}.gif"
        for i in range(len(self.data)):
            with imageio.get_writer(path, mode='I', loop=0) as writer:
                for data in self.data[i]:
                    writer.append_data(data)


class Progress(Callback):
    """Callback to print values during optimization.

    Todo:
        * Write examples
    """
    def on_begin(self, steps=None, **kwargs) -> None:
        self.progbar = tf.keras.utils.Progbar(steps)

    def __call__(self, i, values, grads, scores, model_outputs, regularizations, **kwargs) -> None:
        if len(scores) > 1:
            scores = [(f"Score[{j}]", score_value) for j, score_value in enumerate(scores)]
        else:
            scores = [("Score", score_value) for score_value in scores]
        scores += regularizations
        self.progbar.update(i + 1, scores + regularizations)


@contextmanager
def managed_callbacks(callbacks=None, **kwargs) -> list:
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
                tf.print("Exception args: ", e)
                pass
