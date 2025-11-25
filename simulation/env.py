import gymnasium as gym
from gymnasium import spaces
import numpy as np

class DDoSEnv(gym.Env):
    """
    Custom Environment that follows gym interface.
    Simulates a network link where an Attacker tries to down the service
    and a Defender tries to filter traffic.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(DDoSEnv, self).__init__()

        # --- Constants ---
        self.MAX_CAPACITY = 1000.0  # Mbps
        self.LEGITIMATE_TRAFFIC_MEAN = 300.0
        
        # --- Action Spaces ---
        # Attacker: [Attack Type (0-3), Intensity (0-1), Mutation (0-1)]
        # 0: No Attack, 1: SYN Flood, 2: UDP Flood, 3: HTTP Flood
        # NOW SUPPORTS MULTI-VECTOR: The 'type' is a bitmask or we treat it as primary vector
        # For simplicity in this version, we keep discrete type but add 'Mutation' as a way to blend
        self.attacker_action_space = spaces.Dict({
            "type": spaces.Discrete(4),
            "intensity": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "mutation": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        })

        # Defender: [Threshold (0-1), Filter Focus (0-2), Drop Prob (0-1)]
        # Focus: 0: Source IP, 1: Protocol, 2: Packet Size
        self.defender_action_space = spaces.Dict({
            "threshold": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "focus": spaces.Discrete(3),
            "drop_prob": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        })

        # --- Observation Space ---
        # [Legit Rate, Attack Rate, Protocol Dist (4), Entropy (2)]
        # Normalized to 0-1 roughly
        self.observation_space = spaces.Box(low=0, high=1, shape=(8,), dtype=np.float32)

        self.state = None
        self.steps = 0
        self.max_steps = 100

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        
        # Chaos Mode Flag
        self.chaos_mode = False
        if options and options.get('chaos_mode'):
            self.chaos_mode = True
            
        # Initial state: Normal traffic, no attack
        # Use Poisson for more realistic packet arrival variance
        legit_traffic = np.random.poisson(self.LEGITIMATE_TRAFFIC_MEAN)
        
        self.state = np.array([
            legit_traffic / self.MAX_CAPACITY, # Legit Rate
            0.0,                               # Attack Rate
            0.8, 0.1, 0.05, 0.05,              # Proto Dist (TCP, UDP, ICMP, HTTP) - Normal
            0.9, 0.9                           # Entropy (IP, Size) - High entropy is normal (diverse users)
        ], dtype=np.float32)
        
        return self.state, {}

    def step(self, attacker_action, defender_action):
        self.steps += 1
        
        # 1. Resolve Attacker Action
        attack_type = attacker_action['type']
        intensity = attacker_action['intensity'][0]
        mutation = attacker_action['mutation'][0]

        # Poisson distribution for legitimate traffic
        current_legit = np.random.poisson(self.LEGITIMATE_TRAFFIC_MEAN)
        attack_traffic = 0.0
        
        # Base attack traffic
        if attack_type > 0:
            attack_traffic = intensity * self.MAX_CAPACITY * 1.5 # Can exceed capacity

        # 2. Resolve Defender Action
        threshold = defender_action['threshold'][0] * self.MAX_CAPACITY
        focus = defender_action['focus']
        drop_prob = defender_action['drop_prob'][0]

        # 3. Calculate Traffic Mix & Filtering Logic
        
        defense_effectiveness = 0.0
        collateral_damage = 0.0 # Legitimate traffic dropped

        if attack_type > 0:
            # Ideal defense mapping
            # SYN Flood (1) -> Protocol (1) or IP (0)
            # UDP Flood (2) -> Protocol (1)
            # HTTP Flood (3) -> IP (0) or Size (2) (simplified)
            
            match_score = 0.0
            if attack_type == 1: # SYN
                if focus == 1: match_score = 0.8
                elif focus == 0: match_score = 0.6
            elif attack_type == 2: # UDP
                if focus == 1: match_score = 0.9
            elif attack_type == 3: # HTTP
                if focus == 0: match_score = 0.7
                elif focus == 2: match_score = 0.5
            
            # Mutation lowers match score (Simulating Zero-Day / Obfuscation)
            match_score *= (1.0 - mutation * 0.6) # Increased mutation impact for research difficulty
            
            # Threshold logic
            # If threshold < total_traffic, we start dropping
            # If threshold is close to legit traffic, we are safer
            
            # Simple heuristic for effectiveness based on drop_prob and match
            defense_effectiveness = match_score * drop_prob

            # Collateral damage increases if drop_prob is high but match is low
            # Also increases if threshold is too low (aggressive filtering)
            collateral_damage_rate = drop_prob * (1.0 - match_score) * 0.5
            if threshold < current_legit:
                collateral_damage_rate += 0.2 # Penalty for low threshold
                
            collateral_damage = current_legit * collateral_damage_rate

        else:
            # No attack, but defender might be paranoid
            collateral_damage = current_legit * (drop_prob * 0.1) # Low collateral if no attack but active filtering
            if threshold < current_legit:
                collateral_damage += current_legit * 0.1

        
        blocked_attack = attack_traffic * defense_effectiveness
        final_attack_load = attack_traffic - blocked_attack
        final_legit_load = current_legit - collateral_damage
        
        total_load = final_attack_load + final_legit_load
        
        # 4. Calculate Rewards
        
        # Service Quality (0 to 1)
        # If total load > capacity, service degrades
        overload = max(0, total_load - self.MAX_CAPACITY)
        service_quality = max(0, 1.0 - (overload / self.MAX_CAPACITY))
        
        # Attacker wants: Low Service Quality, High Collateral Damage
        attacker_reward = (1.0 - service_quality) + (collateral_damage / self.MAX_CAPACITY)
        
        # Defender wants: High Service Quality, Low Collateral Damage
        defender_reward = service_quality - (collateral_damage / self.MAX_CAPACITY) * 2.0 # Higher penalty for collateral in research

        # 5. Update State for next step
        # Protocol distribution and entropy shift based on attack type
        new_proto = [0.8, 0.1, 0.05, 0.05] # Default TCP, UDP, ICMP, HTTP
        new_entropy = [0.9, 0.9]
        
        if attack_type == 1: # SYN Flood (TCP)
            new_proto[0] = 0.95 # Mostly TCP
            new_entropy[0] = 0.1 if mutation < 0.5 else 0.8 # Low IP entropy (botnet) unless mutated (spoofed)
        elif attack_type == 2: # UDP Flood
            new_proto[1] = 0.9
        elif attack_type == 3: # HTTP Flood
            new_proto[3] = 0.8
        elif attack_type == 4: # ICMP Flood (Zero-Shot Test Case)
            new_proto[2] = 0.9 # Mostly ICMP
            new_entropy[1] = 0.05 # Low size entropy (ping packets are uniform)
        
        # Noise
        new_proto = np.clip(np.array(new_proto) + np.random.normal(0, 0.05, 4), 0, 1)
        new_entropy = np.clip(np.array(new_entropy) + np.random.normal(0, 0.05, 2), 0, 1)

        # --- Chaos Mode Injection ---
        if self.chaos_mode and np.random.rand() < 0.1: # 10% chance of chaos event
            # Extreme Volatility: Sudden flash crowd or route flap
            # This confuses neural networks that rely on smooth patterns
            final_legit_load *= 3.0 # Flash crowd
            new_entropy[0] = 0.2 # Sudden drop in IP entropy (looks like attack but is just a specific region)
            
        self.state = np.concatenate([
            [final_legit_load / self.MAX_CAPACITY],
            [final_attack_load / self.MAX_CAPACITY],
            new_proto,
            new_entropy
        ], dtype=np.float32)

        terminated = False
        truncated = False
        if self.steps >= self.max_steps:
            truncated = True
            
        info = {
            "legit_load": final_legit_load,
            "attack_load": final_attack_load,
            "collateral": collateral_damage,
            "service_quality": service_quality,
            "attack_type": attack_type
        }

        return self.state, attacker_reward, defender_reward, terminated, truncated, info

    def render(self, mode='human'):
        pass
