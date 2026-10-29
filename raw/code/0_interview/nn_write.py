import numpy as np
import torch
import torch.nn as nn

class simpleNN:

    def __init__(self, input_dim, hidden_dim, output_dim=1, lr = 0.01):
        self.lr = lr 

        W1_init = nn.Linear(input_dim, hidden_dim)
        nn.init.kaiming_normal_(W1_init.weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(W1_init.bias)
        
        self.W1 = W1_init.weight

        W2_init = nn.Linear(hidden_dim, output_dim)
        nn.init.xavier_uniform_(W2_init.weight)
        nn.init.zeros_(W2_init.bias)
        self.W2 = W2_init.weight

    
    def relu()

