from abc import ABC, abstractmethod
from collections import defaultdict

import tensorflow as tf

from tf_keras_vis.utils import listify


class ModelVisualization(ABC):
    """Visualization class for Keras models.
    """
    def __init__(self, model, model_modifier=None, clone=True):
        """Create Visualization class instance that analize the model for debugging.

        # Arguments
            model: The `tf.keras.Model` instance. This model will be cloned by
                `tf.keras.models.clone_model` function and then will be modified by
                `model_modifier` according to need. Therefore the model will be NOT modified.
            model_modifier: A function that modify `model` instance. For example, in
                ActivationMaximization normally, this function is used to replace the softmax
                function that was applied to the model outputs.
            clone: A bool. If you won't model to be copied, you can set this option to False.
        """
        if clone:
            self.model = tf.keras.models.clone_model(model)
            self.model.set_weights(model.get_weights())
        else:
            self.model = model
        if model_modifier is not None:
            new_model = model_modifier(self.model)
            if new_model is not None:
                self.model = new_model

    @abstractmethod
    def __call__(self):
        """Analize the model.

        # Returns
            Results of analizing the model.
        """
        raise NotImplementedError()

    def _prepare_losses(self, loss):
        model_outputs_length = len(self.model.outputs)
        losses = self._prepare_list(loss, model_outputs_length)
        if len(losses) != model_outputs_length:
            raise ValueError('The model has {} outputs, '
                             'but the number of loss functions you passed is {}.'.format(
                                 model_outputs_length, len(losses)))
        return losses

    def _prepare_list(self,
                      value,
                      list_length_if_created,
                      empty_list_if_none=True,
                      convert_tuple_to_list=True):
        values = listify(value,
                         empty_list_if_none=empty_list_if_none,
                         convert_tuple_to_list=convert_tuple_to_list)
        if len(values) == 1 and list_length_if_created > 1:
            values = values * list_length_if_created
        return values

    def _prepare_dictionary(self,
                            values,
                            keys,
                            default_value=list,
                            empty_list_if_none=True,
                            convert_tuple_to_list=True):
        if isinstance(values, dict):
            values = defaultdict(default_value, values)
        else:
            _values = defaultdict(default_value)
            for k in keys:
                _values[k] = values
            values = _values
        for key in values.keys():
            values[key] = listify(values[key],
                                  empty_list_if_none=empty_list_if_none,
                                  convert_tuple_to_list=convert_tuple_to_list)
        return values
