from abc import ABC

import imageio
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont

from tf_keras_vis.utils import normalize


class OptimizerCallback(ABC):
    """Abstract class for defining callbacks.
    """
    def on_begin(self):
        """Called at the begin of optimization process.
        """
        pass

    def __call__(self, i, values, grads, losses, model_outpus, **kwargs):
        """This function will be called within `tf_keras_vis.ModelVisualization` instance.

        # Arguments:
            i: The optimizer iteration.
            values: The current `values`.
            grads: The gradient of input images with respect to `values`.
            losses: A list of loss values with respect to each the model outputs.
            model_outpus: A list of the model outputs.
            kwargs: Optional named arguments that will be used different ways by each
                `tf_keras_vis.ModelVisualization`.
        """
        pass

    def on_end(self):
        """Called at the end of optimization process.
        """
        pass


class Print(OptimizerCallback):
    """Callback to print values during optimization.
    """
    def __init__(self, interval=10):
        """
        # Arguments:
            interval: An integer that appears the interval of printing.
        """
        self.interval = interval

    def __call__(self, i, values, grads, losses, model_outpus, **kwargs):
        i += 1
        if (i % self.interval == 0):
            if 'regularizations' in kwargs:
                tf.print('Steps: {:03d}\tLosses: {},\tRegularizations: {}'.format(
                    i, self._tolist(losses), self._tolist(kwargs['regularizations'])))
            else:
                print('[{:03d}] Losses: {}'.format(i, self._tolist(losses)))

    def _tolist(self, ary):
        if isinstance(ary, list) or isinstance(ary, (np.ndarray, np.generic)):
            return [self._tolist(e) for e in ary]
        elif isinstance(ary, tuple):
            return tuple(self._tolist(e) for e in ary)
        elif tf.is_tensor(ary):
            return ary.numpy().tolist()
        else:
            return ary


class GifGenerator(OptimizerCallback):
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

    def __call__(self, i, values, grads, losses, model_outpus, normalization=False, **kwargs):
        if self.data is None:
            self.data = [[] for i in range(len(values))]
        for n, value in enumerate(values):
            img = Image.fromarray(value[0].astype(np.uint8))  # 1st image in a batch
            ImageDraw.Draw(img).text(
                (10, 10),
                "Step {}".format(i + 1),
                # fill=(0, 0, 0),
                font=ImageFont.load_default())
            if normalization:
                self.data[n].append(normalize(np.asarray(img)))
            else:
                self.data[n].append(np.asarray(img))

    def on_end(self):
        for i in range(len(self.data)):
            path = '{}.{}.gif'.format(self.path, i)
            writer = None
            try:
                writer = imageio.get_writer(path, mode='I', loop=0)
                for data in self.data[i]:
                    writer.append_data(data)
            finally:
                if writer is not None:
                    writer.close()
