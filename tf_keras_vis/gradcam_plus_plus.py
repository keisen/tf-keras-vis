import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

from tf_keras_vis.gradcam import Gradcam


class GradcamPlusPlus(Gradcam):
    def __call__(self,
                 loss,
                 seed_input,
                 penultimate_layer=-1,
                 seek_penultimate_conv_layer=True,
                 activation_modifier=lambda cam: K.relu(cam),
                 normalize_gradient=True):
        """Generate a gradient based class activation map (CAM) by using positive gradient of
            penultimate_layer with respect to loss.

            For details on GradCAM++, see the paper:
            [GradCAM++: Improved Visual Explanations for Deep Convolutional Networks]
            (https://arxiv.org/pdf/1710.11063.pdf).

        # Arguments
            loss: A loss function. If the model has multiple outputs, you can use a different
                loss on each output by passing a list of losses.
            seed_input: An N-dim Numpy array. If the model has multiple inputs,
                you have to pass a list of N-dim Numpy arrays.
            penultimate_layer: A number of integer or a tf.keras.layers.Layer object.
            seek_penultimate_conv_layer: True to seek the penultimate layter that is a subtype of
                `keras.layers.convolutional.Conv` class.
                If False, the penultimate layer is that was elected by penultimate_layer index.
            normalize_gradient: True to normalize gradients.
            activation_modifier: A function to modify gradients.
            expand_cam: True to expand cam to same as input image size.
                ![Note] Even if the model has multiple inputs, this function return only one cam
                value (That's, when `expand_cam` is True, multiple cam images are generated from
                a model that has multiple inputs).
        # Returns
            The heatmap image or a list of their images that indicate the `seed_input` regions
                whose change would most contribute  the loss value,
        # Raises
            ValueError: In case of invalid arguments for `loss`, or `penultimate_layer`.
        """
        # Preparing
        losses = self._get_losses_for_multiple_outputs(loss)
        seed_inputs = self._get_seed_inputs_for_multiple_inputs(seed_input)
        penultimate_output_tensor = self._find_penultimate_output(penultimate_layer,
                                                                  seek_penultimate_conv_layer)
        # Processing gradcam
        model = tf.keras.Model(inputs=self.model.inputs,
                               outputs=self.model.outputs + [penultimate_output_tensor])

        with tf.GradientTape() as tape:
            tape.watch(seed_inputs)
            outputs = model(seed_inputs)
            outputs, penultimate_output = outputs[:-1], outputs[-1]
            loss_values = [loss(y) for y, loss in zip(outputs, losses)]
        grads = tape.gradient(loss_values, penultimate_output)
        if normalize_gradient:
            grads = K.l2_normalize(grads)

        score = sum([K.exp(v) for v in loss_values])
        score = tf.reshape(score, (-1, ) + ((1, ) * (len(grads.shape) - 1)))

        first_derivative = score * grads
        second_derivative = first_derivative * grads
        third_derivative = second_derivative * grads

        global_sum = K.sum(penultimate_output,
                           axis=tuple(np.arange(len(penultimate_output.shape))[1:-1]),
                           keepdims=True)

        alpha_denom = second_derivative * 2.0 + third_derivative * global_sum
        alpha_denom = alpha_denom + tf.cast((second_derivative == 0.0), second_derivative.dtype)
        alphas = second_derivative / alpha_denom

        alpha_normalization_constant = K.sum(alphas,
                                             axis=tuple(np.arange(len(alphas.shape))[1:-1]),
                                             keepdims=True)
        alpha_normalization_constant = alpha_normalization_constant + tf.cast(
            (alpha_normalization_constant == 0.0), alpha_normalization_constant.dtype)
        alphas = alphas / alpha_normalization_constant

        if activation_modifier is None:
            weights = first_derivative
        else:
            weights = activation_modifier(first_derivative)
        deep_linearization_weights = weights * alphas
        deep_linearization_weights = K.sum(
            deep_linearization_weights,
            axis=tuple(np.arange(len(deep_linearization_weights.shape))[1:-1]),
            keepdims=True)

        cam = K.sum(deep_linearization_weights * penultimate_output, axis=-1)
        if activation_modifier is not None:
            cam = activation_modifier(cam)

        cam = self._zoom_for_visualizing(seed_inputs, cam)
        if len(self.model.inputs) == 1 and not isinstance(seed_input, list):
            cam = cam[0]
        return cam
