import pytest

from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.utils.scores import CategoricalScore
from tf_keras_vis.utils.test import dummy_sample


class TestGradcam():
    @pytest.mark.usefixtures("mixed_precision")
    def test__call__if_normalize_gradient_is_True(self, conv_model):
        cam = Gradcam(conv_model)
        result = cam(CategoricalScore(0), dummy_sample((1, 8, 8, 3)), normalize_gradient=True)
        assert result.shape == (1, 8, 8)
