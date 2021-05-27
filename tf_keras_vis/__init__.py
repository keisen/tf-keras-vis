from abc import ABC, abstractmethod

import tensorflow as tf

from tf_keras_vis.utils import listify


class ModelVisualization(ABC):
    """Visualization class for Keras models.

    Attributes:
        model (tf.keras.Model): The target model instance.
    """
    def __init__(self, model, model_modifier=None, clone=True):
        """Create Visualization class instance that analyze the model for debugging.

        Args:
            model (tf.keras.Model): When `model_modifier` is NOT None,
                This model will be cloned by `tf.keras.models.clone_model` function
                and then will be modified by `model_modifier` according to needs.
            model_modifier (function, optional): A function that modify `model` instance.
                For example, in ActivationMaximization usually,
                this function is used to replace the softmax
                function that was applied to the model outputs. Defaults to None.
            clone (bool, optional): When False, the model won't be cloned.
                Note that, although when True, the model won't be clone
                if `model_modifier` is None. Defaults to True.
        """
        self.model = model
        if model_modifier is not None:
            if clone:
                self.model = tf.keras.models.clone_model(self.model)
                self.model.set_weights(model.get_weights())
            new_model = model_modifier(self.model)
            if new_model is not None:
                self.model = new_model

    @abstractmethod
    def __call__(self):
        """Analyze the model.

        Raises:
            NotImplementedError: The `__call__()` of subclass should be called, not this.
        """
        raise NotImplementedError()

    def _get_scores_for_multiple_outputs(self, score):
        scores = listify(score)
        if len(scores) == 1 and len(scores) < len(self.model.outputs):
            scores = scores * len(self.model.outputs)
        for score in scores:
            if not callable(score):
                raise ValueError('Score object must be callable! [{}]'.format(score))
        if len(scores) != len(self.model.outputs):
            raise ValueError(('The model has {} outputs, '
                              'but the number of score-functions you passed is {}.').format(
                                  len(self.model.outputs), len(scores)))
        return scores

    def _get_seed_inputs_for_multiple_inputs(self, seed_input):
        seed_inputs = listify(seed_input)
        if len(seed_inputs) != len(self.model.inputs):
            raise ValueError(('The model has {} inputs, '
                              'but the number of seed-inputs tensors you passed is {}.').format(
                                  len(self.model.inputs), len(seed_inputs)))
        seed_inputs = (x if tf.is_tensor(x) else tf.constant(x) for x in seed_inputs)
        seed_inputs = (tf.expand_dims(x, axis=0) if len(x.shape) == len(tensor.shape[1:]) else x
                       for x, tensor in zip(seed_inputs, self.model.inputs))
        seed_inputs = list(seed_inputs)
        for i, (x, tensor) in enumerate(zip(seed_inputs, self.model.inputs)):
            if len(x.shape) != len(tensor.shape):
                raise ValueError(("seed_input's shape is invalid. model-input index: {},"
                                  " model-input shape: {},"
                                  " seed_input shape: {}.".format(i, tensor.shape, x.shape)))
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
