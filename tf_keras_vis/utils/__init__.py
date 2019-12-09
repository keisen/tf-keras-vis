def listify(value, empty_list_if_none=True, convert_tuple_to_list=True):
    """
    Ensures that the value is a list.
    If it is not a list, it creates a new list with `value` as an item.
    """
    if not isinstance(value, list):
        if value is None and empty_list_if_none:
            value = []
        elif isinstance(value, tuple) and convert_tuple_to_list:
            value = list(value)
        else:
            value = [value]
    return value
