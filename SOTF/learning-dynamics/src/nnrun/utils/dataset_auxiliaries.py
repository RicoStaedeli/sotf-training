import torch


### test_checkpoint_for_nan ##################################################################################################
def test_checkpoint_for_nan(checkpoint, layer_lst, use_bias):
    ddx = []
    for (layer, type) in layer_lst:
        weight = checkpoint.get(f"module_list.{layer}.weight", torch.empty(0))
        bias = checkpoint.get(f"module_list.{layer}.bias", torch.empty(0))
        ddx.append(weight)
        if use_bias:
            ddx.append(bias)
    # check if data has nan
    nan_index = False
    for tens in ddx:
        if torch.isnan(tens).any():
            nan_index = True
    return nan_index


def test_checkpoint_with_threshold(checkpoint, layer_lst, use_bias, threshold):
    if torch.isinf(torch.tensor(threshold)):
        return False

    ddx = []
    for (layer, type) in layer_lst:
        weight = checkpoint.get(f"module_list.{layer}.weight", torch.empty(0))
        bias = checkpoint.get(f"module_list.{layer}.bias", torch.empty(0))
        ddx.append(weight)
        if use_bias:
            ddx.append(bias)
    # check if data has nan
    thresh_index = False
    for tens in ddx:
        tens2 = tens.abs() > threshold
        if tens2.any():
            thresh_index = True
    return thresh_index


def vectorize_checkpoint(checkpoint, layer_lst, use_bias=False):
    # initialize data list
    ddx = []
    # loop over all layers in layer_lst
    for _, (layer, _) in enumerate(layer_lst):
        # load input data and label from data
        weight = checkpoint.get(f"module_list.{layer}.weight", torch.empty(0))
        bias = checkpoint.get(f"module_list.{layer}.bias", torch.empty(0))

        # put weights to be considered in a list for post-processing
        ddx.append(weight)
        if use_bias:
            ddx.append(bias)

    vec = torch.Tensor()
    for idx in ddx:
        vec = torch.cat((vec, idx.view(-1)))
    ddx = vec

    return ddx

