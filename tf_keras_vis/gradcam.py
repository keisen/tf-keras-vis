import numpy as np
import tensorflow as tf
from scipy.ndimage.interpolation import zoom
from tensorflow.keras import backend as K
from tensorflow.python.keras.layers.convolutional import Conv

from tf_keras_vis import ModelVisualization
from tf_keras_vis.utils import find_layer, listify


class Gradcam(ModelVisualization):
    def __call__(self,
                 loss,
                 seed_input,
                 penultimate_layer=-1,
                 activation_modifier=lambda cam: K.relu(cam),
                 normalize_gradient=True):
        """Generate a gradient based class activation map (CAM) by using positive gradient of
            penultimate_layer with respect to loss.

            For details on Grad-CAM, see the paper:
            [Grad-CAM: Why did you say that? Visual Explanations from Deep Networks via
            Gradient-based Localization](https://arxiv.org/pdf/1610.02391v1.pdf).

        # Arguments
            loss: A loss function. If the model has multipul outputs, you can use a different
                loss on each output by passing a list of losses.
            seed_input: An N-dim Numpy array. If the model has multipul inputs,
                you have to pass a list of N-dim Numpy arrays.
            penultimate_layer: A number of integer or a tf.keras.layers.Layer object.
            normalize_gradient: True to normalize gradients.
            activation_modifier: A function to modify gradients.
        # Returns
            The heatmap image or a list of their images that indicate the `seed_input` regions
                whose change would most contribute  the loss value,
        # Raises
            ValueError: In case of invalid arguments for `loss`, or `penultimate_layer`.
        """
        losses = self._prepare_losses(loss)
        seed_inputs = [x if tf.is_tensor(x) else tf.constant(x) for x in listify(seed_input)]
        seed_inputs = [
            tf.expand_dims(seed_input, axis=0) if X.shape == input_tensor.shape[1:] else X
            for X, input_tensor in zip(seed_inputs, self.model.inputs)
        ]
        if len(seed_inputs) != len(self.model.inputs):
            raise ValueError('')

        penultimate_output_tensor = self._find_penultimate_output(self.model, penultimate_layer)
        model = tf.keras.Model(inputs=self.model.inputs,
                               outputs=self.model.outputs + [penultimate_output_tensor])
        with tf.GradientTape() as tape:
            tape.watch(seed_inputs)
            outputs = model(seed_inputs)
            outputs = listify(outputs)
            loss_values = [loss(y) for y, loss in zip(outputs[:-1], losses)]
            penultimate_outputs = outputs[-1]
        grads = tape.gradient(loss_values, penultimate_outputs)
        if normalize_gradient:
            grads = K.l2_normalize(grads)
        weights = K.mean(grads, axis=tuple(np.arange(len(grads.shape))[1:-1]))
        cam = np.asarray([np.sum(o * w, axis=-1) for o, w in zip(penultimate_outputs, weights)])
        if activation_modifier is not None:
            cam = activation_modifier(cam)
        input_dims_list = (X.shape[1:-1] for X in seed_inputs)
        output_dims = penultimate_outputs.shape[1:-1]
        zoom_factors = ([i / (j * 1.0) for i, j in iter(zip(input_dims, output_dims))]
                        for input_dims in input_dims_list)
        cams = [np.asarray([zoom(v, factor) for v in cam]) for factor in zoom_factors]
        if len(self.model.inputs) == 1 and not isinstance(seed_input, list):
            cams = cams[0]
        return cams

    def _find_penultimate_output(self, model, layer):
        if not isinstance(layer, tf.keras.layers.Layer):
            if layer is None:
                layer = -1
            if isinstance(layer, int) and layer < len(model.layers):
                layer = model.layers[int(layer)]
            elif isinstance(layer, str):
                layer = find_layer(model, lambda l: l.name == layer)
            else:
                raise ValueError('Invalid argument. `layer`=', layer)
        if layer is not None:
            layer = find_layer(model, lambda l: isinstance(l, Conv), offset=layer)
        if layer is None:
            raise ValueError('Unable to determine penultimate `Conv` layer.')
        return layer.output
