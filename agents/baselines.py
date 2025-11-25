import numpy as np

class RandomDefender:
    def __init__(self):
        pass
        
    def get_action(self, state):
        # Random actions
        return {
            "threshold": np.random.rand(1).astype(np.float32),
            "focus": np.random.randint(0, 3),
            "drop_prob": np.random.rand(1).astype(np.float32)
        }

class StaticThresholdDefender:
    def __init__(self, threshold=0.8):
        self.threshold = threshold
        
    def get_action(self, state):
        # Always sets a fixed threshold and moderate filtering
        return {
            "threshold": np.array([self.threshold], dtype=np.float32),
            "focus": 1, # Default to Protocol filtering
            "drop_prob": np.array([0.5], dtype=np.float32)
        }
