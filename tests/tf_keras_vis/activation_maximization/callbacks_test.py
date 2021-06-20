import os

import pytest

from tf_keras_vis.activation_maximization import \
    ActivationMaximization as CurrentActivationMaximization  # noqa: E501
from tf_keras_vis.activation_maximization.callbacks import GifGenerator2D, PrintLogger, Progress
from tf_keras_vis.activation_maximization.legacy import \
    ActivationMaximization as LegacyActivationMaximization  # noqa: E501
from tf_keras_vis.utils.scores import CategoricalScore

ActivationMaximization = CurrentActivationMaximization


@pytest.fixture(scope='function',
                params=[CurrentActivationMaximization, LegacyActivationMaximization])
def legacy(request):
    global ActivationMaximization
    ActivationMaximization = request.param
    yield
    ActivationMaximization = CurrentActivationMaximization


class TestPrintLogger():
    def test__init__(self):
        interval = 999
        logger = PrintLogger(interval)
        assert logger.interval == interval

    @pytest.mark.usefixtures("mixed_precision", "legacy")
    def test__call__(self, conv_model):
        activation_maximization = ActivationMaximization(conv_model)
        result = activation_maximization(CategoricalScore(1), steps=1, callbacks=PrintLogger(1))
        assert result.shape == (1, 8, 8, 3)

    @pytest.mark.usefixtures("mixed_precision", "legacy")
    def test__call__without_regularization(self, conv_model):
        activation_maximization = ActivationMaximization(conv_model)
        result = activation_maximization(CategoricalScore(1),
                                         steps=1,
                                         regularizers=None,
                                         callbacks=PrintLogger(1))
        assert result.shape == (1, 8, 8, 3)


class TestProgress():
    @pytest.mark.usefixtures("mixed_precision", "legacy")
    def test__call__(self, multiple_outputs_model):
        activation_maximization = ActivationMaximization(multiple_outputs_model)
        result = activation_maximization(
            [CategoricalScore(0), CategoricalScore(0)], callbacks=Progress())
        assert result.shape == (1, 8, 8, 3)

    @pytest.mark.usefixtures("mixed_precision", "legacy")
    def test__call__without_regularizers(self, conv_model):
        activation_maximization = ActivationMaximization(conv_model)
        result = activation_maximization(CategoricalScore(0),
                                         regularizers=None,
                                         callbacks=Progress())
        assert result.shape == (1, 8, 8, 3)


class TestGifGenerator2D():
    def test__init__(self, tmpdir):
        path = tmpdir.mkdir("tf-keras-vis").join("test.gif")
        generator = GifGenerator2D(path)
        assert generator.path == path

    @pytest.mark.usefixtures("mixed_precision", "legacy")
    def test__call__(self, tmpdir, conv_model):
        path = tmpdir.mkdir("tf-keras-vis").join("test.gif")
        activation_maximization = ActivationMaximization(conv_model)
        assert not os.path.isfile(path)
        result = activation_maximization(CategoricalScore(0), callbacks=GifGenerator2D(str(path)))
        assert os.path.isfile(path)
        assert result.shape == (1, 8, 8, 3)
