import os

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

MAX_STEPS = 'TF_KERAS_VIS_MAX_STEPS'


def check_steps(steps):
    """
    Load max-steps value that is for avoiding timeout on travis-ci when testing Notebook.
    """
    max_steps = int(os.environ[MAX_STEPS]) if MAX_STEPS in os.environ else steps
    return min(steps, max_steps)


def num_of_gpus():
    if tf.__version__.startswith('2.0.'):
        list_physical_devices = tf.config.experimental.list_physical_devices
        list_logical_devices = tf.config.experimental.list_logical_devices
    else:
        list_physical_devices = tf.config.list_physical_devices
        list_logical_devices = tf.config.list_logical_devices
    physical_gpus = list_physical_devices('GPU')
    if physical_gpus:
        logical_gpus = list_logical_devices('GPU')
        return len(physical_gpus), len(logical_gpus)
    else:
        return 0, 0


def listify(value, return_empty_list_if_none=True, convert_tuple_to_list=True):
    """
    Ensures that the value is a list.
    If it is not a list, it creates a new list with `value` as an item.

    # Arguments
        value: A list or something else.
        empty_list_if_none: A boolean. When True (default), None you passed as `value` will be
            converted to a empty list (i.e., `[]`). But when False, it will be converted to a list
            that has an None (i.e., `[None]`)
        convert_tuple_to_list: A boolean. When True (default), a tuple you passed as `value` will be
            converted to a list. But when False, it will be unconverted (i.e., returning a tuple
            object that was passed as `value`).
    # Returns
        A list, but when `value` is a tuple and `convert_tuple_to_list` is False, a tuple.
    """
    if not isinstance(value, list):
        if value is None and return_empty_list_if_none:
            value = []
        elif isinstance(value, tuple) and convert_tuple_to_list:
            value = list(value)
        else:
            value = [value]
    return value


def normalize(array, value_range=(1., 0.)):
    max_value = np.max(array, axis=tuple(range(array.ndim)[1:]), keepdims=True)
    min_value = np.min(array, axis=tuple(range(array.ndim)[1:]), keepdims=True)
    normalized_array = (array - min_value) / (max_value - min_value + K.epsilon())
    if value_range is None:
        return normalized_array
    else:
        high, low = value_range
        return (high - low) * normalized_array + low


def find_layer(model, condition, offset=None, reverse=True):
    found_offset = offset is None
    for layer in reversed(model.layers):
        if not found_offset:
            found_offset = (layer == offset)
        if condition(layer) and found_offset:
            return layer
        if isinstance(layer, tf.keras.Model):
            if found_offset:
                result = find_layer(layer, condition, offset=None, reverse=reverse)
            else:
                result = find_layer(layer, condition, offset=offset, reverse=reverse)
            if result is not None:
                return result
    return None


def zoom_factor(from_shape, to_shape):
    return tuple(t / f for f, t in iter(zip(from_shape, to_shape)))
