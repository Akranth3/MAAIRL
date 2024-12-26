# import torch

# '''
# Prints a short summaary of the model, input, hidden units, output units, and total parameters.
# '''

# def summary_model(model):
#     print("Model Summary")
#     last = None
#     print("the flow of the model is input--->")
#     for i in model.children():
#         print(i.__class__.__name__)
#         print( i.in_features, "----->")
#         last = i.out_features
#     print( last)

import torch
import torch.nn as nn
from typing import List, Tuple
import textwrap

def get_activation_name(layer: nn.Module) -> str:
    """Get the name of the activation function."""
    activation_map = {
        nn.ReLU: 'ReLU',
        nn.LeakyReLU: 'LeakyReLU',
        nn.Tanh: 'Tanh',
        nn.Sigmoid: 'Sigmoid',
        nn.GELU: 'GELU',
        nn.ELU: 'ELU',
        nn.Softmax: 'Softmax'
    }
    return activation_map.get(type(layer), '')

def is_activation_layer(layer: nn.Module) -> bool:
    """Check if the layer is an activation function."""
    return isinstance(layer, (nn.ReLU, nn.LeakyReLU, nn.Tanh, nn.Sigmoid, 
                            nn.GELU, nn.ELU, nn.Softmax))

def count_parameters(model: nn.Module) -> int:
    """Count the total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def format_number(num: int) -> str:
    """Format large numbers with commas."""
    return f"{num:,}"

def summary_model(model: nn.Module, input_size: Tuple[int, ...] = None) -> None:
    """
    Print a detailed summary of the PyTorch model architecture.
    
    Args:
        model (nn.Module): PyTorch model to summarize
        input_size (tuple, optional): Size of input tensor (excluding batch dimension)
    """
    # Terminal width for formatting
    term_width = 80
    separator = "=" * term_width
    
    # Header
    print(f"\n{separator}")
    print(f"{'Model Summary':^{term_width}}")
    print(f"{separator}\n")
    
    # Architecture flow
    layers = []
    current_size = list(input_size) if input_size else None
    
    for name, layer in model.named_modules():
        if isinstance(layer, (nn.Linear, nn.Conv2d)) or is_activation_layer(layer):
            layer_info = {}
            
            if isinstance(layer, nn.Linear):
                layer_info['type'] = 'Linear'
                layer_info['in_features'] = layer.in_features
                layer_info['out_features'] = layer.out_features
            elif isinstance(layer, nn.Conv2d):
                layer_info['type'] = 'Conv2d'
                layer_info['in_channels'] = layer.in_channels
                layer_info['out_channels'] = layer.out_channels
                layer_info['kernel_size'] = layer.kernel_size
            else:
                layer_info['type'] = get_activation_name(layer)
            
            layers.append(layer_info)
    
    # Print architecture flow
    print("Architecture Flow:")
    print("-" * term_width)
    
    if input_size:
        flow = f"({' × '.join(map(str, input_size))})"
    else:
        flow = ""
        
    for i, layer in enumerate(layers):
        if layer['type'] in ['Linear', 'Conv2d']:
            if layer['type'] == 'Linear':
                flow += f" → [{layer['type']}] → ({layer['out_features']})"
            else:
                kernel_size = f"{layer['kernel_size']}k" if isinstance(layer['kernel_size'], int) else f"{layer['kernel_size'][0]}k"
                flow += f" → [{layer['type']}{kernel_size}] → ({layer['out_channels']})"
        else:
            flow += f" → [{layer['type']}]"
    
    # Print wrapped flow
    wrapped_flow = textwrap.fill(flow, width=term_width-2)
    for line in wrapped_flow.split('\n'):
        print(f"  {line}")
    
    # Print summary statistics
    print(f"\n{separator}")
    print("Summary Statistics:")
    print("-" * term_width)
    print(f"Total Parameters: {format_number(count_parameters(model))}")
    print(f"Layers: {len([l for l in layers if l['type'] in ['Linear', 'Conv2d']])}")
    print(f"Activation Functions: {len([l for l in layers if l['type'] not in ['Linear', 'Conv2d']])}")
    print(f"{separator}\n")

# Example usage
"""
model = nn.Sequential(
    nn.Linear(784, 512),
    nn.ReLU(),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)
summary_model(model, input_size=(784,))
"""
