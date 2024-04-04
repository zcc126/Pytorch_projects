import torch
import torch.nn as nn
import torch.nn.functional as F

class FlexibleCNN(nn.Module):
    def __init__(self, input_channels, conv_params, fc_params):
        super().__init__()
        self.input_channels = input_channels
        self.conv_params = conv_params
        self.fc_params = fc_params
        
        # Define convolutional layers
        self.conv_layers = nn.ModuleList()
        in_channels = input_channels
        for params in conv_params:
            out_channels, kernel_size, padding = params
            self.conv_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding))
            in_channels = out_channels
        
        # Calculate the size of the output from convolutional layers
        self.conv_out_size = self._calculate_conv_output_size()
        
        # Define fully connected layers
        self.fc_layers = nn.ModuleList()
        in_features = self.conv_out_size
        for out_features in fc_params:
            self.fc_layers.append(nn.Linear(in_features, out_features))
            in_features = out_features

    def _calculate_conv_output_size(self):
        # Forward a dummy input through convolutional layers to calculate output size
        x = torch.zeros(1, self.input_channels, 28, 28)
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
            x = F.relu(x)
            x = F.max_pool2d(x, kernel_size=2, stride=2)
        return x.view(x.size(0), -1).size(1)

    def forward(self, x):
        # Forward pass through convolutional layers
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
            x = F.relu(x)
            x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(-1, self.conv_out_size)
        
        # Forward pass through fully connected layers
        for fc_layer in self.fc_layers[:-1]:
            x = F.relu(fc_layer(x))
        x = self.fc_layers[-1](x)
        return x
