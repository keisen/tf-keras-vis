import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K


def listify(value, empty_list_if_none=True, convert_tuple_to_list=True):
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
        if value is None and empty_list_if_none:
            value = []
        elif isinstance(value, tuple) and convert_tuple_to_list:
            value = list(value)
        else:
            value = [value]
    return value


def normalize(array, value_range=(1., 0.)):
    max_value = np.max(array)
    min_value = np.min(array)
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
