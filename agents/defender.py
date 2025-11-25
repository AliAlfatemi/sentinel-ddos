from .base_agent import GeneticAgent
import numpy as np

class DefenderAgent(GeneticAgent):
    def __init__(self, input_dim=8, output_dim=5, hidden_dim=64):
        # Input: 8 (State)
        # Output: 5 (1 Threshold, 3 Focus Logits, 1 Drop Prob)
        super().__init__(input_dim, output_dim, hidden_dim)
        
    def get_action(self, state):
        out = self.forward(state)
        
        # 1. Threshold (0-1)
        threshold = 1.0 / (1.0 + np.exp(-out[0])) # Sigmoid
        
        # 2. Focus (0-2)
        focus_logits = out[1:4]
        focus = np.argmax(focus_logits)
        
        # 3. Drop Prob (0-1)
        drop_prob = 1.0 / (1.0 + np.exp(-out[4])) # Sigmoid
        
        return {
            "threshold": np.array([threshold], dtype=np.float32),
            "focus": focus,
            "drop_prob": np.array([drop_prob], dtype=np.float32)
        }
