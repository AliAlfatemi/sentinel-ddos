import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import pickle

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.env import DDoSEnv
from agents.neuro_symbolic import SentinelAgent
from agents.attacker import AttackerAgent

def run_adaptation_test(agent):
    env = DDoSEnv()
    state, _ = env.reset()
    
    # Scenario: Switch attacks every 30 steps
    # 0-30: Type 1 (SYN)
    # 30-60: Type 2 (UDP)
    # 60-90: Type 3 (HTTP)
    # 90-120: Type 4 (ICMP - Zero Shot)
    
    history = {
        "step": [],
        "service_quality": [],
        "attack_type": []
    }
    
    # We need a dummy attacker that we can force
    attacker = AttackerAgent()
    
    print("Running Adaptation Speed Test...")
    
    for step in range(120):
        # Determine Attack Type
        if step < 30: attack_type = 1
        elif step < 60: attack_type = 2
        elif step < 90: attack_type = 3
        else: attack_type = 4
        
        # Force Attacker Action
        att_action = attacker.get_action(state)
        att_action['type'] = attack_type
        att_action['intensity'] = np.array([1.0], dtype=np.float32) # Full power
        
        # Defender Action
        def_action = agent.get_action(state)
        
        next_state, _, _, _, _, info = env.step(att_action, def_action)
        state = next_state
        
        history["step"].append(step)
        history["service_quality"].append(info['service_quality'])
        history["attack_type"].append(attack_type)
        
    return pd.DataFrame(history)

def run_heatmap_analysis(agent):
    env = DDoSEnv()
    attacker = AttackerAgent()
    
    print("Running Action Distribution Analysis...")
    
    data = {
        "Attack Type": [],
        "Defender Focus": []
    }
    
    focus_map = {0: "Source IP", 1: "Protocol", 2: "Packet Size"}
    attack_map = {1: "SYN Flood", 2: "UDP Flood", 3: "HTTP Flood", 4: "ICMP Flood"}
    
    # Run 50 steps for each attack type
    for type_id in [1, 2, 3, 4]:
        state, _ = env.reset()
        for _ in range(50):
            att_action = attacker.get_action(state)
            att_action['type'] = type_id
            att_action['intensity'] = np.array([1.0], dtype=np.float32)
            
            def_action = agent.get_action(state)
            
            data["Attack Type"].append(attack_map[type_id])
            data["Defender Focus"].append(focus_map[def_action['focus']])
            
            state, _, _, _, _, _ = env.step(att_action, def_action)
            
    return pd.DataFrame(data)

def main():
    # Load Best Agent (Sentinel)
    sentinel = SentinelAgent()
    if os.path.exists("best_defender_hof.pkl"):
        with open("best_defender_hof.pkl", "rb") as f:
            base_neural = pickle.load(f)
            sentinel.W1 = base_neural.W1
            sentinel.b1 = base_neural.b1
            sentinel.W2 = base_neural.W2
            sentinel.b2 = base_neural.b2
    else:
        print("Warning: No trained model found. Using random weights.")

    # 1. Adaptation Test
    df_adapt = run_adaptation_test(sentinel)
    df_adapt.to_csv("adaptation_results.csv", index=False)
    
    # Plot Adaptation
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=df_adapt, x="step", y="service_quality", linewidth=2.5)
    
    # Add vertical lines for switches
    plt.axvline(x=30, color='r', linestyle='--', alpha=0.5, label='Attack Switch')
    plt.axvline(x=60, color='r', linestyle='--', alpha=0.5)
    plt.axvline(x=90, color='r', linestyle='--', alpha=0.5)
    
    # Add text labels
    plt.text(15, 0.5, "SYN", ha='center', fontsize=12, fontweight='bold')
    plt.text(45, 0.5, "UDP", ha='center', fontsize=12, fontweight='bold')
    plt.text(75, 0.5, "HTTP", ha='center', fontsize=12, fontweight='bold')
    plt.text(105, 0.5, "ICMP (Zero-Shot)", ha='center', fontsize=12, fontweight='bold')
    
    plt.title("Sentinel Adaptation Speed (Multi-Vector Scenario)")
    plt.ylabel("Service Quality")
    plt.xlabel("Simulation Step")
    plt.ylim(0, 1.1)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("adaptation_plot.png")
    print("Saved adaptation_plot.png")

    # 2. Heatmap Analysis
    df_heat = run_heatmap_analysis(sentinel)
    
    # Create Contingency Table
    ct = pd.crosstab(df_heat["Attack Type"], df_heat["Defender Focus"], normalize='index')
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(ct, annot=True, cmap="Blues", fmt=".2f", cbar_kws={'label': 'Selection Probability'})
    plt.title("Defender Strategy Distribution")
    plt.tight_layout()
    plt.savefig("strategy_heatmap.png")
    print("Saved strategy_heatmap.png")

if __name__ == "__main__":
    main()
