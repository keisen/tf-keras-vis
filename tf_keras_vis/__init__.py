from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import tensorflow as tf

from .utils import listify


class ModelVisualization(ABC):
    """Visualization class that analyze the model for debugging.
    """
    def __init__(self, model, model_modifier=None, clone=True) -> None:
        """
        Args:
            model: A  `tf.keras.Model` instance. When `model_modifier` is NOT None, this model will
                be cloned with `tf.keras.models.clone_model` function and then will be modified by
                `model_modifier` according to needs.
            model_modifier: A :obj:`tf_keras_vis.utils.model_modifiers.ModelModifier` instance,
                a function or a list of them. We recommend to apply
                `tf_keras_vis.utils.model_modifiers.ReplaceToLinear` to all visualizations (except
                :obj:`tf_keras_vis.scorecam.Scorecam`) when the model output is softmax. Defaults
                to None.
            clone: A bool that indicates whether or not it clones the `model`. When False, the
                model won't be cloned. Note that, although when True, the model won't be clone if
                `model_modifier` is None. Defaults to True.
        """
        self.model = model
        model_modifiers = listify(model_modifier)
        if len(model_modifiers) > 0:
            if clone:
                self.model = tf.keras.models.clone_model(self.model)
                self.model.set_weights(model.get_weights())
            for modifier in model_modifiers:
                new_model = modifier(self.model)
                if new_model is not None:
                    self.model = new_model

    @abstractmethod
    def __call__(self) -> Union[np.ndarray, list]:
        """Analyze the model.

        Raises:
            NotImplementedError: This method must be overwritten.

        Returns:
            Visualized image(s) or something(s).
        """
        raise NotImplementedError()

    def _get_scores_for_multiple_outputs(self, score):
        scores = listify(score)
        for score in scores:
            if not callable(score):
                raise ValueError(f"Score object must be callable! [{score}]")
        if len(scores) != len(self.model.outputs):
            raise ValueError(f"The model has {len(self.model.outputs)} outputs, "
                             f"but the number of score-functions you passed is {len(scores)}.")
        return scores

    def _get_seed_inputs_for_multiple_inputs(self, seed_input):
        seed_inputs = listify(seed_input)
        if len(seed_inputs) != len(self.model.inputs):
            raise ValueError(
                f"The model has {len(self.model.inputs)} inputs, "
                f"but the number of seed-inputs tensors you passed is {len(seed_inputs)}.")
        seed_inputs = (x if tf.is_tensor(x) else tf.constant(x) for x in seed_inputs)
        seed_inputs = (tf.expand_dims(x, axis=0) if len(x.shape) == len(tensor.shape[1:]) else x
                       for x, tensor in zip(seed_inputs, self.model.inputs))
        seed_inputs = list(seed_inputs)
        for i, (x, tensor) in enumerate(zip(seed_inputs, self.model.inputs)):
            if len(x.shape) != len(tensor.shape):
                raise ValueError(
                    f"seed_input's shape is invalid. model-input index: {i},"
                    f" model-input shape: {tensor.shape}, seed_input shape: {x.shape}.")
        return seed_inputs

    def _calculate_scores(self, outputs, score_functions):
        score_values = (func(output) for output, func in zip(outputs, score_functions))
        score_values = (self._mean_score_value(score) for score in score_values)
        score_values = list(score_values)
        return score_values

    def _mean_score_value(self, score):
        if not tf.is_tensor(score):
            if type(score) in [list, tuple]:
                if len(score) > 0 and tf.is_tensor(score[0]):
                    score = tf.stack(score, axis=0)
                else:
                    score = tf.constant(score)
            else:
                score = tf.constant(score)
        score = tf.math.reduce_mean(score, axis=tuple(range(score.ndim))[1:])
        return score
