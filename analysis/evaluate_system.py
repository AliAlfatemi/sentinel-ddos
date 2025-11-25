import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import sys
import pickle

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.env import DDoSEnv
from agents.attacker import AttackerAgent
from agents.defender import DefenderAgent
from agents.baselines import RandomDefender, StaticThresholdDefender

def run_experiment(defender_type="co-evolutionary", episodes=50):
    env = DDoSEnv()
    
    # Load best attacker to test against (or create a new evolving one)
    # For fair comparison, we test against a standard evolving attacker
    attacker = AttackerAgent()
    
    if defender_type == "co-evolutionary":
        if os.path.exists("best_defender.pkl"):
            with open("best_defender.pkl", "rb") as f:
                defender = pickle.load(f)
        else:
            print("No trained model found, using fresh agent")
            defender = DefenderAgent()
    elif defender_type == "random":
        defender = RandomDefender()
    elif defender_type == "static":
        defender = StaticThresholdDefender()
        
    results = {
        "episode": [],
        "service_quality": [],
        "collateral_damage": [],
        "mitigation_efficiency": [],
        "defender_type": []
    }
    
    for ep in range(episodes):
        state, _ = env.reset()
        ep_quality = 0
        ep_collateral = 0
        ep_attack_load = 0
        ep_blocked = 0
        
        for _ in range(100):
            att_action = attacker.get_action(state)
            def_action = defender.get_action(state)
            
            # Attacker learns online during evaluation to simulate adaptive threat
            # But for baselines, we might want a fixed attacker or also adaptive?
            # Let's keep attacker adaptive to show which defender handles adaptation better
            
            next_state, att_r, def_r, term, trunc, info = env.step(att_action, def_action)
            
            state = next_state
            ep_quality += info['service_quality']
            ep_collateral += info['collateral']
            ep_attack_load += info['attack_load']
            
            # Simple online learning for attacker to make it a threat
            if isinstance(attacker, AttackerAgent):
                # Simple mutation-based hill climbing for evaluation
                if att_r > 0.5: # If successful, keep parameters slightly mutated
                    attacker.mutate(0.1)
                else:
                    attacker.mutate(0.3) # Explore more
            
            if term or trunc:
                break
        
        # Metrics
        avg_quality = ep_quality / 100
        total_collateral = ep_collateral
        
        # Efficiency: Quality / (1 + Collateral)
        efficiency = avg_quality / (1.0 + total_collateral / 1000.0)
        
        results["episode"].append(ep)
        results["service_quality"].append(avg_quality)
        results["collateral_damage"].append(total_collateral)
        results["mitigation_efficiency"].append(efficiency)
        results["defender_type"].append(defender_type)
        
    return pd.DataFrame(results)

def main():
    print("Running Comprehensive Evaluation...")
    
    # 1. Test Co-Evolutionary Agent
    print("Testing Co-Evolutionary Agent...")
    df_co = run_experiment("co-evolutionary", episodes=100)
    
    # 2. Test Random Baseline
    print("Testing Random Baseline...")
    df_rand = run_experiment("random", episodes=100)
    
    # 3. Test Static Baseline
    print("Testing Static Baseline...")
    df_static = run_experiment("static", episodes=100)
    
    # Combine
    full_df = pd.concat([df_co, df_rand, df_static])
    full_df.to_csv("evaluation_results.csv", index=False)
    print("Evaluation Complete. Saved to evaluation_results.csv")

if __name__ == "__main__":
    main()
