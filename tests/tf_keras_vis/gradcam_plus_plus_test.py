import pytest
import tensorflow.keras.backend as K

from tf_keras_vis.gradcam_plus_plus import GradcamPlusPlus
from tf_keras_vis.utils.scores import CategoricalScore
from tf_keras_vis.utils.test import dummy_sample


class TestGradcamPlusPlus():
    @pytest.mark.parametrize("activation_modifier", [None, (lambda cam: K.relu(cam))])
    @pytest.mark.usefixtures("mixed_precision")
    def test__call__if_activation_modifier_is_(self, activation_modifier, conv_model):
        cam = GradcamPlusPlus(conv_model)
        result = cam(CategoricalScore(0),
                     dummy_sample((1, 8, 8, 3)),
                     activation_modifier=activation_modifier)
        assert result.shape == (1, 8, 8)
