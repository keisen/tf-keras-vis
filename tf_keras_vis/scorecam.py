import numpy as np
from scipy.ndimage.interpolation import zoom
import tensorflow as tf
import tensorflow.keras.backend as K

from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.utils import zoom_factor


class ScoreCAM(Gradcam):
    def __init__(self, model):
        """Create ScoreCAM class instance that analize the model for debugging.

        # Arguments
            model: The `tf.keras.Model` instance.
        """
        super().__init__(model, model_modifier=None, clone=False)
        if len(model.outputs) > 1:
            raise ValueError(("`model` has multiple outputs but ,currently, "
                              "ScoreCAM doesn't yet support such a model."
                              "If you needed, please feel free to request it on Github-Issues."))
        if model.layers[-1].activation != tf.keras.activations.softmax:
            raise ValueError(("`model` MUST has a output layer "
                              "that is set a softmax activation function."))

    def __call__(self,
                 loss,
                 seed_input,
                 penultimate_layer=-1,
                 seek_penultimate_conv_layer=True,
                 activation_modifier=lambda cam: K.relu(cam),
                 normalize_gradient=True,
                 expand_cam=True):

        # Preparing
        losses = self._get_losses_for_multiple_outputs(loss)  # TODO
        seed_inputs = self._get_seed_inputs_for_multiple_inputs(seed_input)
        penultimate_output_tensor = self._find_penultimate_output(penultimate_layer,
                                                                  seek_penultimate_conv_layer)
        # Processing score-cam
        model = tf.keras.Model(inputs=self.model.inputs, outputs=penultimate_output_tensor)
        penultimate_output = model(seed_inputs)

        # Resizing activation-maps
        factors = (zoom_factor(penultimate_output.shape, seed_input.shape)
                   for seed_input in seed_inputs)
        resized_activation_maps = [zoom(penultimate_output, factor) for factor in factors]

        # Normalizing activation-maps
        min_activation_maps = (np.min(activation_map,
                                      axis=tuple(range(len(activation_map.shape))[1:-1]),
                                      keepdims=True) for activation_map in resized_activation_maps)
        max_activation_maps = (np.max(activation_map,
                                      axis=tuple(range(len(activation_map.shape))[1:-1]),
                                      keepdims=True) for activation_map in resized_activation_maps)
        normalized_activation_maps = [
            (activation_map - min_activation_map) / (max_activation_map - min_activation_map)
            for activation_map, min_activation_map, max_activation_map in zip(
                resized_activation_maps, min_activation_maps, max_activation_maps)
        ]

        # Masking input-sheeds by multiply by activation-maps.
        input_tile_axes = (
            (activation_map.shape[-1], ) + tuple(np.ones(len(seed_input.shape), np.int))
            for seed_input, activation_map in zip(seed_inputs, normalized_activation_maps))
        map_tile_axes = (
            (seed_input.shape[-1], ) + tuple(np.ones(len(activation_map.shape), np.int))
            for seed_input, activation_map in zip(seed_inputs, normalized_activation_maps))
        map_transpose_axes = ((len(activation_map.shape), ) +
                              tuple(range(len(activation_map.shape))[1:]) + (0, )
                              for activation_map in normalized_activation_maps)
        mask_templates = (np.tile(seed_input, axes)
                          for seed_input, axes in zip(seed_inputs, input_tile_axes))
        masks = (np.transpose(np.tile(activation_map, tile_axis),
                              transpose_axis) for activation_map, tile_axis, transpose_axis in zip(
                                  normalized_activation_maps, map_tile_axes, map_transpose_axes))
        masked_seed_inputs = [
            np.reshape(mask_template * mask, (-1, ) + mask.shape[2:])
            for mask_template, mask in zip(mask_templates, masks)
        ]

        # Predicting masked seed-inputs
        preds = self.model.predict(masked_seed_inputs)
        if losses == 1:
            preds = [preds]
        preds = [
            np.reshape(prediction, (penultimate_output.shape[-1], -1, prediction.shape[-1]))
            for prediction in preds
        ]

        # Calculating weights
        weights = [[loss(p) for p in prediction] for loss, prediction in zip(losses, preds)]
        weights = np.sum(np.array(weights), axis=0)
        weights = np.transpose(weights, (1, 0))
        if len(penultimate_output.shape) > 2:
            weights = np.reshape(weights, (weights.shape[0], ) +
                                 tuple(np.ones(len(penultimate_output.shape[1:-1]))) +
                                 (weights.shape[-1], ))

        # Generate cam
        cam = penultimate_output * weights
        cam = activation_modifier(cam)
        cam /= np.max(cam)

        if not expand_cam:
            return cam

        factors = (zoom_factor(cam.shape, X.shape) for X in seed_inputs)
        cam = [zoom(cam, factor) for factor in factors]
        if len(self.model.inputs) == 1 and not isinstance(seed_input, list):
            cam = cam[0]
        return cam
