from typing import Union

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

from .gradcam import Gradcam


class Layercam(Gradcam):
    """LayerCAM

    References:
        * LayerCAM: Exploring Hierarchical Class Activation Maps for Localization
          (https://ieeexplore.ieee.org/document/9462463)
    """
    def __call__(self,
                 score,
                 seed_input,
                 penultimate_layer=None,
                 seek_penultimate_conv_layer=True,
                 gradient_modifier=lambda grads: K.relu(grads),
                 activation_modifier=lambda cam: K.relu(cam),
                 training=False,
                 expand_cam=True,
                 standardize_cam=True,
                 unconnected_gradients=tf.UnconnectedGradients.NONE) -> Union[np.ndarray, list]:
        arguments = locals().items()
        arguments = ((k, v) for k, v in arguments if k != 'self')
        arguments = ((k, v) for k, v in arguments if not k.startswith('_'))
        arguments = dict(arguments)
        return super().__call__(**arguments)
