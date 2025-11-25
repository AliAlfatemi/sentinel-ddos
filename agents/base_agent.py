import numpy as np
import copy

class GeneticAgent:
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        
        # Initialize weights
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2/input_dim)
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2/hidden_dim)
        self.b2 = np.zeros(output_dim)
        
    def forward(self, state):
        # Forward pass
        x = state
        h = np.tanh(np.dot(x, self.W1) + self.b1)
        out = np.dot(h, self.W2) + self.b2
        return out
    
    def mutate(self, mutation_rate=0.01, mutation_scale=0.1):
        # Add noise to weights
        if np.random.rand() < mutation_rate:
            self.W1 += np.random.randn(*self.W1.shape) * mutation_scale
        if np.random.rand() < mutation_rate:
            self.b1 += np.random.randn(*self.b1.shape) * mutation_scale
        if np.random.rand() < mutation_rate:
            self.W2 += np.random.randn(*self.W2.shape) * mutation_scale
        if np.random.rand() < mutation_rate:
            self.b2 += np.random.randn(*self.b2.shape) * mutation_scale
            
    def clone(self):
        # Return a deep copy
        new_agent = self.__class__(self.input_dim, self.output_dim, self.hidden_dim)
        new_agent.W1 = self.W1.copy()
        new_agent.b1 = self.b1.copy()
        new_agent.W2 = self.W2.copy()
        new_agent.b2 = self.b2.copy()
        return new_agent
