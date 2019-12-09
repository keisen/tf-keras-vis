from abc import ABC

import imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


class Callback(ABC):
    def __init__(self):
        pass

    def on_begin(self):
        pass

    def __call__(self, i, values, grads, losses, model_outpus, **kwargs):
        pass

    def on_end(self):
        pass


class Print(Callback):
    def __init__(self, interval=10):
        self.interval = interval

    def __call__(self, i, values, grads, losses, model_outpus, **kwargs):
        if (i % self.interval == 0):
            if 'regularizations' in kwargs:
                print('[{:03d}] Losses: {}, Regularization: {}'.format(
                    i, [np.asarray(ary)[0] for ary in losses],
                    [np.asarray(ary) for ary in kwargs['regularizations']]))
            else:
                print('[{:03d}] Losses: {}'.format(i, [np.asarray(ary)[0] for ary in losses]))


class GifGenerator(Callback):
    def __init__(self, path):
        self.path = path

    def on_begin(self):
        self.data = None

    def __call__(self, i, values, losses, grads, regularization, outputs):
        if self.data is None:
            self.data = [[] for i in range(len(values))]
        for n, value in enumerate(values):
            img = Image.fromarray(value[0].astype(np.uint8))  # 1st image in a batch
            ImageDraw.Draw(img).text((10, 10),
                                     "Step {}".format(i + 1),
                                     fill=(0, 0, 0),
                                     font=ImageFont.load_default())
            self.data[n].append(np.asarray(img))

    def on_end(self):
        for i in range(len(self.data)):
            path = '{}.{}.gif'.format(self.path, i)
            writer = None
            try:
                writer = imageio.get_writer(path, mode='I', loop=1)
                for data in self.data[i]:
                    writer.append_data(data)
            finally:
                if writer is not None:
                    writer.close()
