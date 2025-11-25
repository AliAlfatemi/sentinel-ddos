from .defender import DefenderAgent
from sklearn.tree import DecisionTreeClassifier, export_text
import numpy as np

class SentinelAgent(DefenderAgent):
    def __init__(self):
        super().__init__()
        self.safety_violations = 0
        self.rule_extractor = DecisionTreeClassifier(max_depth=3) # Simple tree for interpretability
        self.memory_states = []
        self.memory_actions = []
        
    def get_action(self, state):
        # 1. Neural Action
        neural_action = super().get_action(state)
        
        # 2. Symbolic Safety Layer (The "Sentinel")
        safe_action = self._enforce_safety(state, neural_action)
        
        # 3. Store for Rule Crystallization
        self.memory_states.append(state)
        # We store the discrete focus for classification
        self.memory_actions.append(safe_action['focus'])
        
        return safe_action
    
    def _enforce_safety(self, state, action):
        """
        Symbolic Logic to override dangerous neural actions.
        """
        threshold = action['threshold'][0]
        focus = action['focus']
        drop_prob = action['drop_prob'][0]
        
        legit_load = state[0]
        
        # Safety Rule 1: Never drop > 80% traffic if legit load is high (>0.5)
        # This prevents "Panic Blocking" which causes outages
        if drop_prob > 0.8 and legit_load > 0.5:
            self.safety_violations += 1
            # Override: Cap drop probability
            action['drop_prob'] = np.array([0.5], dtype=np.float32)
            
        # Safety Rule 2: If Entropy is High (Normal), don't focus on Source IP
        # High IP entropy means distributed traffic (or normal users). Blocking IPs here is bad.
        ip_entropy = state[6]
        if ip_entropy > 0.8 and focus == 0: # Focus 0 is Source IP
            self.safety_violations += 1
            # Override: Switch focus to Protocol (1) which is safer
            action['focus'] = 1
            
        return action

    def crystallize_rules(self):
        """
        Extracts IF-THEN rules from the agent's recent behavior.
        """
        if len(self.memory_states) < 50:
            return "Not enough data to crystallize rules."
            
        # Fit Decision Tree
        X = np.array(self.memory_states)
        y = np.array(self.memory_actions)
        
        self.rule_extractor.fit(X, y)
        
        # Export Rules
        feature_names = ["LegitRate", "AttackRate", "TCP", "UDP", "ICMP", "HTTP", "EntIP", "EntSize"]
        rules = export_text(self.rule_extractor, feature_names=feature_names)
        
        # Clear memory to keep it fresh
        self.memory_states = []
        self.memory_actions = []
        
        return rules
