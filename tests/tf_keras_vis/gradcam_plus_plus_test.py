import pytest
import tensorflow as tf

from tf_keras_vis import keras
from tf_keras_vis.gradcam_plus_plus import GradcamPlusPlus
from tf_keras_vis.utils.scores import CategoricalScore
from tf_keras_vis.utils.test import dummy_sample


class TestGradcamPlusPlus():
    @pytest.mark.parametrize("gradient_modifier",
                             [None, (lambda cam: keras.activations.relu(cam))])
    @pytest.mark.parametrize("activation_modifier",
                             [None, (lambda cam: keras.activations.relu(cam))])
    @pytest.mark.usefixtures("mixed_precision")
    def test__call__if_activation_modifier_is_(self, gradient_modifier, activation_modifier,
                                               conv_model):
        cam = GradcamPlusPlus(conv_model)
        result = cam(CategoricalScore(0),
                     dummy_sample((1, 8, 8, 3)),
                     gradient_modifier=gradient_modifier,
                     activation_modifier=activation_modifier)
        assert result.shape == (1, 8, 8)
