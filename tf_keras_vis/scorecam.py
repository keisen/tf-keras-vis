import numpy as np
from scipy.ndimage.interpolation import zoom
import tensorflow as tf
import tensorflow.keras.backend as K

from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.utils import listify, zoom_factor


class ScoreCAM(Gradcam):
    def __call__(self,
                 loss,
                 seed_input,
                 penultimate_layer=-1,
                 seek_penultimate_conv_layer=True,
                 activation_modifier=lambda cam: K.relu(cam),
                 expand_cam=True,
                 batch_size=32,
                 max_N=None,
                 training=False):
        """Generate score-weighted class activation maps (CAM) by using gradient-free visualization method.

            For details on Score-CAM, see the paper:
            [Score-CAM: Score-Weighted Visual Explanations for Convolutional Neural Networks ]
            (https://arxiv.org/pdf/1910.01279.pdf).

        # Arguments
            loss: A loss function. If the model has multiple outputs, you can use a different
                loss on each output by passing a list of losses.
            seed_input: An N-dim Numpy array. If the model has multiple inputs,
                you have to pass a list of N-dim Numpy arrays.
            penultimate_layer: A number of integer or a tf.keras.layers.Layer object.
            seek_penultimate_conv_layer: True to seek the penultimate layter that is a subtype of
                `keras.layers.convolutional.Conv` class.
                If False, the penultimate layer is that was elected by penultimate_layer index.
            activation_modifier: A function to modify activations.
            expand_cam: True to expand cam to same as input image size.
                ![Note] Even if the model has multiple inputs, this function return only one cam
                value (That's, when `expand_cam` is True, multiple cam images are generated from
                a model that has multiple inputs).
            batch_size: Integer or None. Number of samples per batch.
                If unspecified, batch_size will default to 32.
            max_N: Integer or None. If None, we do NOT recommend, because it takes huge time.
                If not None, that's setting Integer, run as Faster-ScoreCAM.
                Set larger number, need more time to visualize CAM but to be able to get
                clearer attention images.
                (see for details: https://github.com/tabayashi0117/Score-CAM#faster-score-cam)
            training: A bool whether the model's trainig-mode turn on or off.
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
        # Processing score-cam
        penultimate_output = tf.keras.Model(inputs=self.model.inputs,
                                            outputs=penultimate_output_tensor)(seed_inputs,
                                                                               training=training)
        # For efficiently visualizing, extract maps that has a large variance.
        # This excellent idea is devised by tabayashi0117.
        # (see for details: https://github.com/tabayashi0117/Score-CAM#faster-score-cam)
        if max_N is not None and max_N > -1:
            activation_map_std = tf.math.reduce_std(penultimate_output,
                                                    axis=tuple(
                                                        range(penultimate_output.ndim)[1:-1]),
                                                    keepdims=True)
            _, top_k_indices = tf.math.top_k(activation_map_std, max_N)
            top_k_indices, _ = tf.unique(tf.reshape(top_k_indices, (-1, )))
            penultimate_output = tf.gather(penultimate_output, top_k_indices, axis=-1)
        channels = penultimate_output.shape[-1]

        # Upsampling activation-maps
        penultimate_output = penultimate_output.numpy()
        input_shapes = [seed_input.shape for seed_input in seed_inputs]
        factors = (zoom_factor(penultimate_output.shape[:-1], input_shape[:-1])
                   for input_shape in input_shapes)
        upsampled_activation_maps = [zoom(penultimate_output, factor + (1, )) for factor in factors]
        map_shapes = [activation_map.shape for activation_map in upsampled_activation_maps]

        # Normalizing activation-maps
        min_activation_maps = (np.min(activation_map,
                                      axis=tuple(range(activation_map.ndim)[1:-1]),
                                      keepdims=True)
                               for activation_map in upsampled_activation_maps)
        max_activation_maps = (np.max(activation_map,
                                      axis=tuple(range(activation_map.ndim)[1:-1]),
                                      keepdims=True)
                               for activation_map in upsampled_activation_maps)
        normalized_activation_maps = (
            (activation_map - min_activation_map) / (max_activation_map - min_activation_map)
            for activation_map, min_activation_map, max_activation_map in zip(
                upsampled_activation_maps, min_activation_maps, max_activation_maps))

        # Masking inputs
        input_tile_axes = ((map_shape[-1], ) + tuple(np.ones(len(input_shape), np.int))
                           for input_shape, map_shape in zip(input_shapes, map_shapes))
        mask_templates = (np.tile(seed_input, axes)
                          for seed_input, axes in zip(seed_inputs, input_tile_axes))
        map_transpose_axes = ((len(map_shape) - 1, ) + tuple(range(len(map_shape))[:-1])
                              for map_shape in map_shapes)
        masks = (np.transpose(activation_map,
                              transpose_axis) for activation_map, transpose_axis in zip(
                                  normalized_activation_maps, map_transpose_axes))
        map_tile_axes = (tuple(np.ones(len(map_shape), np.int)) + (input_shape[-1], )
                         for input_shape, map_shape in zip(input_shapes, map_shapes))
        masks = (np.tile(np.expand_dims(activation_map, axis=-1), tile_axis)
                 for activation_map, tile_axis in zip(masks, map_tile_axes))
        masked_seed_inputs = (mask_template * mask
                              for mask_template, mask in zip(mask_templates, masks))
        masked_seed_inputs = [
            np.reshape(masked_seed_input, (-1, ) + masked_seed_input.shape[2:])
            for masked_seed_input in masked_seed_inputs
        ]

        # Predicting masked seed-inputs
        preds = self.model.predict(masked_seed_inputs, batch_size=batch_size)
        preds = (np.reshape(prediction, (channels, -1, prediction.shape[-1]))
                 for prediction in listify(preds))

        # Calculating weights
        weights = ([loss(p) for p in prediction] for loss, prediction in zip(losses, preds))
        weights = (np.array(w, dtype=np.float32) for w in weights)
        weights = (np.reshape(w, (channels, -1)) for w in weights)
        weights = np.array(list(weights), dtype=np.float32)
        weights = np.sum(weights, axis=0)
        weights = np.transpose(weights, (1, 0))

        # Generate cam
        cam = K.batch_dot(penultimate_output, weights)
        if activation_modifier is not None:
            cam = activation_modifier(cam)

        if not expand_cam:
            return cam

        factors = (zoom_factor(cam.shape, X.shape) for X in seed_inputs)
        cam = [zoom(cam, factor) for factor in factors]
        if len(self.model.inputs) == 1 and not isinstance(seed_input, list):
            cam = cam[0]
        return cam
