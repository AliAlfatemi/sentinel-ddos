from .base_agent import GeneticAgent
import numpy as np

class AttackerAgent(GeneticAgent):
    def __init__(self, input_dim=8, output_dim=6, hidden_dim=64):
        # Input: 8 (State)
        # Output: 6 (4 Type Logits, 1 Intensity, 1 Mutation)
        super().__init__(input_dim, output_dim, hidden_dim)
        
    def get_action(self, state):
        out = self.forward(state)
        
        # 1. Attack Type (0-3)
        type_logits = out[:4]
        attack_type = np.argmax(type_logits)
        
        # 2. Intensity (0-1)
        intensity = 1.0 / (1.0 + np.exp(-out[4])) # Sigmoid
        
        # 3. Mutation (0-1)
        mutation = 1.0 / (1.0 + np.exp(-out[5])) # Sigmoid
        
        return {
            "type": attack_type,
            "intensity": np.array([intensity], dtype=np.float32),
            "mutation": np.array([mutation], dtype=np.float32)
        }
