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
from analysis.xai_explainer import XAIExplainer

def evaluate(env, attacker, defender, episodes=5, force_attack_type=None):
    total_att_reward = 0
    total_def_reward = 0
    total_quality = 0
    
    for _ in range(episodes):
        state, _ = env.reset()
        for _ in range(100): # Max steps
            att_action = attacker.get_action(state)
            
            # Force attack type for Zero-Shot testing
            if force_attack_type is not None:
                att_action['type'] = force_attack_type
                
            def_action = defender.get_action(state)
            
            next_state, att_r, def_r, term, trunc, info = env.step(att_action, def_action)
            
            state = next_state
            total_att_reward += att_r
            total_def_reward += def_r
            total_quality += info['service_quality']
            
            if term or trunc:
                break
                
    return total_att_reward / episodes, total_def_reward / episodes, total_quality / (episodes * 100)

def train():
    env = DDoSEnv()
    explainer = XAIExplainer()
    
    # Hyperparameters
    POP_SIZE = 20
    GENERATIONS = 2000 # Massive Scale
    ELITE_SIZE = 4
    MUTATION_RATE = 0.1
    HOF_INTERVAL = 50 # Save to Hall of Fame every 50 gens
    
    # Initialize Populations
    attackers = [AttackerAgent() for _ in range(POP_SIZE)]
    defenders = [DefenderAgent() for _ in range(POP_SIZE)]
    
    # Hall of Fame
    hall_of_fame_attackers = []
    
    results = {
        "generation": [],
        "best_attacker_reward": [],
        "best_defender_reward": [],
        "avg_service_quality": [],
        "curriculum_phase": []
    }
    
    print(f"Starting Massive Scale HoF Training (Pop: {POP_SIZE}, Gens: {GENERATIONS})...")
    
    for gen in tqdm(range(GENERATIONS)):
        # --- Curriculum Logic ---
        curriculum_phase = "Phase 1: Simple"
        if gen >= 50: curriculum_phase = "Phase 2: Multi-Vector"
        if gen >= 100: curriculum_phase = "Phase 3: Full Mutation"
        
        att_scores = np.zeros(POP_SIZE)
        def_scores = np.zeros(POP_SIZE)
        gen_quality = 0
        
        # Evaluate
        for i in range(POP_SIZE):
            # Defender Evaluation: Play against current population AND Hall of Fame
            opponents = []
            
            # 1. Current Population Opponents (Standard Co-Evolution)
            current_opps = np.random.choice(POP_SIZE, 3, replace=False)
            for idx in current_opps:
                opponents.append(attackers[idx])
                
            # 2. Hall of Fame Opponents (Robustness Check)
            # 20% chance to face a historical ghost if HoF exists
            if len(hall_of_fame_attackers) > 0 and np.random.rand() < 0.2:
                hof_attacker = hall_of_fame_attackers[np.random.randint(len(hall_of_fame_attackers))]
                opponents.append(hof_attacker)
            
            for attacker in opponents:
                # Note: We are evaluating defenders[i] against multiple attackers
                # We also need to score the attackers, but HoF attackers don't get scores (they are frozen)
                
                # To keep scoring simple for the co-evolution part:
                # We only update att_scores for the CURRENT population attackers.
                # We update def_scores for ALL matches.
                
                att_r, def_r, qual = evaluate(env, attacker, defenders[i], episodes=1)
                
                def_scores[i] += def_r
                gen_quality += qual
                
                # If the attacker is from the current population, update its score
                # We need to find if 'attacker' is in our 'attackers' list
                # This is a bit inefficient, so let's just reverse the loop structure slightly
                pass

        # Optimized Evaluation Loop
        # Reset scores
        att_scores = np.zeros(POP_SIZE)
        def_scores = np.zeros(POP_SIZE)
        gen_quality = 0
        
        # Each Defender plays 3 matches against Current Pop + potentially 1 against HoF
        for def_idx in range(POP_SIZE):
            # Match 1-3: Current Population
            att_indices = np.random.choice(POP_SIZE, 3, replace=False)
            for att_idx in att_indices:
                att_r, def_r, qual = evaluate(env, attackers[att_idx], defenders[def_idx], episodes=1)
                att_scores[att_idx] += att_r
                def_scores[def_idx] += def_r
                gen_quality += qual
            
            # Match 4: Hall of Fame (Optional)
            if len(hall_of_fame_attackers) > 0 and np.random.rand() < 0.2:
                hof_att = hall_of_fame_attackers[np.random.randint(len(hall_of_fame_attackers))]
                _, def_r, qual = evaluate(env, hof_att, defenders[def_idx], episodes=1)
                def_scores[def_idx] += def_r # Bonus points for beating history
                # Don't update gen_quality to keep it comparable to standard runs, or do?
                # Let's keep it for robustness metric.
        
        # Normalize scores roughly
        # Attackers played approx 3 times each (on average)
        # Defenders played 3 + epsilon times
        att_scores /= 3
        def_scores /= 3.2 # Approx normalization
        
        best_att_idx = np.argmax(att_scores)
        best_def_idx = np.argmax(def_scores)
        
        # Update Hall of Fame
        if gen % HOF_INTERVAL == 0:
            # Save a clone of the best attacker
            hall_of_fame_attackers.append(attackers[best_att_idx].clone())
        
        results["generation"].append(gen)
        results["best_attacker_reward"].append(att_scores[best_att_idx])
        results["best_defender_reward"].append(def_scores[best_def_idx])
        results["avg_service_quality"].append(gen_quality / (POP_SIZE * 3))
        results["curriculum_phase"].append(curriculum_phase)
        
        if gen % 100 == 0:
            print(f"Gen {gen} ({curriculum_phase}): Best Def R: {def_scores[best_def_idx]:.2f} | HoF Size: {len(hall_of_fame_attackers)}")
            
        # Selection & Reproduction
        att_sorted = np.argsort(att_scores)[::-1]
        def_sorted = np.argsort(def_scores)[::-1]
        
        new_attackers = []
        new_defenders = []
        
        for i in range(ELITE_SIZE):
            new_attackers.append(attackers[att_sorted[i]].clone())
            new_defenders.append(defenders[def_sorted[i]].clone())
            
        while len(new_attackers) < POP_SIZE:
            parent = attackers[np.random.choice(att_sorted[:ELITE_SIZE])]
            child = parent.clone()
            child.mutate(mutation_rate=MUTATION_RATE)
            new_attackers.append(child)
            
        while len(new_defenders) < POP_SIZE:
            parent = defenders[np.random.choice(def_sorted[:ELITE_SIZE])]
            child = parent.clone()
            child.mutate(mutation_rate=MUTATION_RATE)
            new_defenders.append(child)
            
        attackers = new_attackers
        defenders = new_defenders

    # Save Results
    df = pd.DataFrame(results)
    df.to_csv("training_results_hof.csv", index=False)
    
    # Save Best Models
    best_defender = defenders[best_def_idx]
    with open("best_defender_hof.pkl", "wb") as f:
        pickle.dump(best_defender, f)

    # --- Zero-Shot Generalization Test ---
    print("\n--- Running Zero-Shot Generalization Test (ICMP Flood) ---")
    dummy_attacker = AttackerAgent() 
    zs_att_r, zs_def_r, zs_qual = evaluate(env, dummy_attacker, best_defender, episodes=50, force_attack_type=4)
    print(f"Zero-Shot Performance (ICMP Flood): Service Quality = {zs_qual:.2f}")
    
    # --- XAI Demo ---
    print("\n--- Generating XAI Explanations ---")
    state, _ = env.reset()
    env.step({'type': 4, 'intensity': [1.0], 'mutation': [0.0]}, {'threshold': [0.5], 'focus': 0, 'drop_prob': [0.0]})
    action = best_defender.get_action(env.state)
    explanation = explainer.explain(env.state, action, attack_type_ground_truth=4)
    print(f"XAI Output: {explanation}")
    
    with open("xai_report.txt", "w") as f:
        f.write(f"Zero-Shot ICMP Quality: {zs_qual:.2f}\n")
        f.write(f"Example Explanation:\n{explanation}\n")


if __name__ == "__main__":
    train()
