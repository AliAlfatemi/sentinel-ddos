import numpy as np

class XAIExplainer:
    def __init__(self):
        pass
        
    def explain(self, state, action, attack_type_ground_truth=None):
        """
        Generates a natural language explanation for the Defender's action
        based on the current network state.
        """
        # State: [Legit Rate, Attack Rate, TCP, UDP, ICMP, HTTP, Entropy IP, Entropy Size]
        # Action: [Threshold, Focus, Drop Prob]
        
        threshold = action['threshold'][0]
        focus = action['focus']
        drop_prob = action['drop_prob'][0]
        
        # 1. Analyze Action Severity
        severity = "Monitoring"
        if drop_prob > 0.8: severity = "Blocking Hard"
        elif drop_prob > 0.4: severity = "Filtering"
        elif drop_prob > 0.1: severity = "Inspecting"
        
        if severity == "Monitoring":
            return "Traffic is normal. No active mitigation required."
            
        # 2. Analyze Reason (State Features)
        reasons = []
        
        # Check Protocol Anomalies
        tcp_ratio = state[2]
        udp_ratio = state[3]
        icmp_ratio = state[4]
        http_ratio = state[5]
        
        if tcp_ratio > 0.9: reasons.append("abnormally high TCP traffic (potential SYN Flood)")
        if udp_ratio > 0.5: reasons.append("suspicious UDP spike")
        if icmp_ratio > 0.5: reasons.append("unusual ICMP volume (Ping Flood)")
        if http_ratio > 0.5: reasons.append("HTTP request surge")
        
        # Check Entropy (Botnet detection)
        entropy_ip = state[6]
        entropy_size = state[7]
        
        if entropy_ip < 0.3: reasons.append("low source IP entropy (likely botnet/spoofed)")
        if entropy_size < 0.3: reasons.append("uniform packet sizes (automated script)")
        
        # 3. Construct Explanation
        focus_str = ["Source IP", "Protocol", "Packet Size"][focus]
        
        explanation = f"Action: {severity} (Focus: {focus_str}). "
        
        if reasons:
            explanation += "Reason: Detected " + ", ".join(reasons) + "."
        else:
            explanation += "Reason: Precautionary filtering due to high overall load."
            
        # 4. Ground Truth Check (for evaluation/debugging)
        if attack_type_ground_truth is not None and attack_type_ground_truth > 0:
            explanation += f" [Actual Attack: Type {attack_type_ground_truth}]"
            
        return explanation
