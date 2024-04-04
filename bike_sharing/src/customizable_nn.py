import torch.nn as nn
import torch.nn.functional as F

class FlexibleNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, activations):
        super().__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.activations = activations
        
        self.relu_layer = nn.ReLU()
        self.sigmoid_layer = nn.Sigmoid()
        self.tanh_layer = nn.Tanh()

        # Create hidden and output layers
        self.layers = nn.ModuleList()
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))

    def forward(self, x):
        # forward propagation
        for i, layer in enumerate(self.layers):
            
            x = x.float()
            x = layer(x)
            
            # Apply activation function
            if i < len(self.layers) - 1:
                activation = self.activations[i]
                if activation == 'relu':
                    x = self.relu_layer(x)
                elif activation == 'sigmoid':
                    x = self.sigmoid_layer(x)
                elif x == 'tanh':
                    x = self.tanh_layer(x)
                else:
                    raise ValueError("Unsupported activation function")
        return x