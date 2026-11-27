import torch
import torch.nn as nn
import math
from typing import List


class Solution:

    def xavier_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        # Return a (fan_out x fan_in) weight matrix using Xavier/Glorot normal initialization
        # Use torch.manual_seed(0) for reproducibility
        # Round to 4 decimal places and return as nested list

        torch.manual_seed(0)

        ## weight matrix is scaled down by sqrt(2/(fan_in+fan_out))
        xavier_std = math.sqrt(2/(fan_in + fan_out))

        xavier_weights = torch.randn(fan_out, fan_in) * xavier_std

        return torch.round(xavier_weights, decimals=4).tolist()

    def kaiming_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        # Return a (fan_out x fan_in) weight matrix using Kaiming/He normal initialization (for ReLU)
        # Use torch.manual_seed(0) for reproducibility
        # Round to 4 decimal places and return as nested list

        torch.manual_seed(0)

        kaiming_std = math.sqrt(2 / fan_in)

        kaiming_weights = torch.randn(fan_out, fan_in) * kaiming_std

        return torch.round(kaiming_weights, decimals=4).tolist()

    def check_activations(self, num_layers: int, input_dim: int, hidden_dim: int, init_type: str) -> List[float]:
        # Use torch.manual_seed(0) once at the start.
        # Sample all weights in layer order with torch.randn(out_dim, in_dim) * std:
        # first (hidden_dim, input_dim), then (hidden_dim, hidden_dim).
        # Keep weights unrounded; do not call the init helpers, which reset the seed.
        # Then sample x = torch.randn(1, input_dim) without resetting the seed.
        # Apply x = torch.relu(x @ W.T) at every layer, with no biases.
        # Return the std of activations after each layer, rounded to 2 decimals.
        # Use round(x.std().item(), 2) with PyTorch's default sample std.

        torch.manual_seed(0)
        
        W = []

        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            out_dim = hidden_dim

            if init_type == 'xavier':
                std = math.sqrt(2/(in_dim + out_dim))
            elif init_type == 'kaiming':
                std = math.sqrt((2/in_dim))
            else:
                std = 1.0

            m = torch.randn(out_dim, in_dim) * std
            W.append(m)

        
        x = torch.randn(1, input_dim)

        results = []

        for i in range(num_layers):
            x = torch.relu(x @ W[i].T)
            results.append(round(x.std().item(), 2))



        return results
