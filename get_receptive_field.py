import torch
import torch.nn as nn
import torch.nn.functional as F

def compute_receptive_field(network, target_layer_id=None):
    """
    Compute the receptive field of a source layer as seen in a target layer.
    :param network: The PyTorch model
    :param target_layer_id: The ID of the target layer
    :return: The receptive field of the source layer in the target layer
    """
    def extract_conv_and_upsample_layers(model):
        layers = []
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                layers.append(('conv', module.kernel_size, module.stride, module.padding))
            elif isinstance(module, nn.Upsample):
                layers.append(('upsample', module.scale_factor))
        return layers

    # Extract the convolution and upsampling layers
    layers = extract_conv_and_upsample_layers(network)
    
    # Compute receptive field
    receptive_field = 1
    current_stride = 1

    
    for i, layer in enumerate(layers):
        if target_layer_id is not None and i == target_layer_id + 1:
            break
        if layer[0] == 'conv':
            k, s, p = layer[1], layer[2], layer[3]
            receptive_field += (k[0] - 1) * current_stride
            current_stride *= s[0]
        elif layer[0] == 'upsample':
            scale_factor = layer[1]
            current_stride //= scale_factor
        print(f"Layer {i}: {layer}, RF: {int(receptive_field)}")

    return receptive_field