
def _build_loss(layer, filter_indices):
    layer_output = layer.output

    # For all other layers it is 4
    is_dense = K.ndim(layer_output) == 2

    loss = 0.
    for idx in listify(filter_indices):
        if is_dense:
            loss += -K.mean(layer_output[:, idx])
        else:
            loss += -K.mean(layer_output[..., idx])
    return loss
    

def visualize_activation(model, layer_idx, filter_indices=None, seed_input=None, input_range=(0, 255), max_iter=200, callbacks=None,verbose=True):
    return ActivationMaximization(model)(_build_loss(model.layers[layer_idx], filter_indices), seed_input=seed_input, input_range=input_range, steps=max_iter, callbacks=callbacks)
