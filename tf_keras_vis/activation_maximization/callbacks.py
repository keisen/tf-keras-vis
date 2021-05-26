from abc import ABC

import imageio
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont


class Callback(ABC):
    """Abstract class for defining callbacks.
    """
    def on_begin(self):
        """Called at the begin of optimization process.
        """
        pass

    def __call__(self, i, values, grads, score_values, outputs, regularizer_values, overall_score):
        """This function will be called within
            `tf_keras_vis.activation_maximization.ActivationMaximization` instance.

        # Arguments:
            i: The optimizer iteration.
            values: The current `values`.
            grads: The gradient of input images with respect to `values`.
            scores: A list of score values with respect to each the model outputs.
            model_outputs: A list of the model outputs.
            kwargs: Optional named arguments that will be used different ways by each
                `tf_keras_vis.activation_maximization.ActivationMaximization`.
        """
        pass

    def on_end(self):
        """Called at the end of optimization process.
        """
        pass


class PrintLogger(Callback):
    """Callback to print values during optimization.
    """
    def __init__(self, interval=10):
        """
        # Arguments:
            interval: An integer that appears the interval of printing.
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
    """
    def __init__(self, path):
        """
        # Arguments:
            path: The file path to save gif.
        """
        self.path = path

    def on_begin(self):
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
