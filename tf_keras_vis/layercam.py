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
                 normalize_cam=True,
                 unconnected_gradients=tf.UnconnectedGradients.NONE) -> Union[np.ndarray, list]:
        """Generate gradient based class activation maps (CAM) by using positive gradient of
        penultimate_layer with respect to score.

        Args:
            score: A :obj:`tf_keras_vis.utils.scores.Score` instance, function or a list of them.
                For example of the Score instance to specify visualizing target::

                    scores = CategoricalScore([1, 294, 413])

                The code above means the same with the one below::

                    score = lambda outputs: (outputs[0][1], outputs[1][294], outputs[2][413])

                When the model has multiple outputs, you MUST pass a list of
                Score instances or functions. For example::

                    from tf_keras_vis.utils.scores import CategoricalScore, InactiveScore
                    score = [
                        CategoricalScore([1, 23]),  # For 1st model output
                        InactiveScore(),            # For 2nd model output
                        ...
                    ]

            seed_input: A tf.Tensor, :obj:`numpy.ndarray` or a list of them to input in the model.
                That's when the model has multiple inputs, you MUST pass a list of tensors.
            penultimate_layer: An index or name of the layer, or the tf.keras.layers.Layer
                instance itself. When None, it means the same with `-1`. If the layer specified by
                this option is not `convolutional` layer, `penultimate_layer` will work as the
                offset to seek `convolutional` layer. Defaults to None.
            seek_penultimate_conv_layer: A bool that indicates whether or not seeks a penultimate
                layer when the layer specified by `penultimate_layer` is not `convolutional` layer.
                Defaults to True.
            activation_modifier: A function to modify the Class Activation Map (CAM). Defaults to
                `lambda cam: K.relu(cam)`.
            training: A bool that indicates whether the model's training-mode on or off. Defaults
                to False.
            gradient_modifier: A function to modify gradients. Defaults to
                `lambda grads: tf.keras.backend.relu(grads)`.
            expand_cam: True to resize CAM to the same as input image size. **Note!** When False,
                even if the model has multiple inputs, return only a CAM. Defaults to True.
            normalize_cam: When True, CAM will be normalized. Defaults to True.
            unconnected_gradients: Specifies the gradient value returned when the given input
                tensors are unconnected. Defaults to tf.UnconnectedGradients.NONE.

        Returns:
            An :obj:`numpy.ndarray` or a list of them. They are the Class Activation Maps (CAMs)
            that indicate the `seed_input` regions whose change would most contribute the score
            value.

        Raises:
            :obj:`ValueError`: When there is any invalid arguments.
        """
        arguments = locals().items()
        arguments = ((k, v) for k, v in arguments if k != 'self')
        arguments = ((k, v) for k, v in arguments if not k.startswith('_'))
        arguments = dict(arguments)
        return super().__call__(**arguments)

    def _calculate_cam(self, grads, penultimate_output, gradient_modifier, activation_modifier):
        if gradient_modifier is not None:
            grads = gradient_modifier(grads)
        cam = np.sum(np.multiply(penultimate_output, grads), axis=-1)
        if activation_modifier is not None:
            cam = activation_modifier(cam)
        return cam
