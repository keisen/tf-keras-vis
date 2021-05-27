import os

from tf_keras_vis.activation_maximization import ActivationMaximization
from tf_keras_vis.activation_maximization.callbacks import GifGenerator2D, PrintLogger
from tf_keras_vis.utils.test import MockScore


class TestPrintLogger():
    def test__init__(self):
        interval = 999
        logger = PrintLogger(interval)
        assert logger.interval == interval

    def test__call__(self, conv_model):
        activation_maximization = ActivationMaximization(conv_model)
        result = activation_maximization(MockScore(), steps=1, callbacks=PrintLogger(1))
        assert result.shape == (1, 8, 8, 3)

    def test__call__without_regularizers(self, conv_model):
        activation_maximization = ActivationMaximization(conv_model)
        result = activation_maximization(MockScore(),
                                         steps=1,
                                         regularizers=None,
                                         callbacks=PrintLogger(1))
        assert result.shape == (1, 8, 8, 3)


class TestGifGenerator2D():
    def test__init__(self, tmpdir):
        path = tmpdir.mkdir("tf-keras-vis").join("test.gif")
        generator = GifGenerator2D(path)
        assert generator.path == path

    def test__call__(self, tmpdir, conv_model):
        path = tmpdir.mkdir("tf-keras-vis").join("test.gif")
        activation_maximization = ActivationMaximization(conv_model)
        assert not os.path.isfile(path)
        result = activation_maximization(MockScore(), steps=1, callbacks=GifGenerator2D(str(path)))
        assert os.path.isfile(path)
        assert result.shape == (1, 8, 8, 3)
