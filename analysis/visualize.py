import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_results():
    # 1. Training Plot (if exists)
    if os.path.exists("training_results.csv"):
        df_train = pd.read_csv("training_results.csv")
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.plot(df_train['generation'], df_train['best_attacker_reward'], label='Best Attacker Reward', alpha=0.7)
        plt.plot(df_train['generation'], df_train['best_defender_reward'], label='Best Defender Reward', alpha=0.7)
        plt.xlabel('Generation')
        plt.ylabel('Reward')
        plt.title('Co-Evolutionary Arms Race')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(df_train['generation'], df_train['avg_service_quality'], color='green', label='Service Quality')
        plt.xlabel('Generation')
        plt.ylabel('Quality (0-1)')
        plt.title('Service Resilience')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig("training_plot.png")
        print("Training plot saved.")

    # 2. Evaluation Plot (Comparative)
    if os.path.exists("evaluation_results.csv"):
        df_eval = pd.read_csv("evaluation_results.csv")
        
        plt.figure(figsize=(14, 6))
        
        # Service Quality Comparison
        plt.subplot(1, 3, 1)
        sns.boxplot(x='defender_type', y='service_quality', data=df_eval)
        plt.title('Service Quality Distribution')
        plt.ylabel('Quality (Higher is Better)')
        
        # Collateral Damage Comparison
        plt.subplot(1, 3, 2)
        sns.barplot(x='defender_type', y='collateral_damage', data=df_eval)
        plt.title('Avg Collateral Damage')
        plt.ylabel('Dropped Legit Packets (Lower is Better)')
        
        # Mitigation Efficiency
        plt.subplot(1, 3, 3)
        sns.violinplot(x='defender_type', y='mitigation_efficiency', data=df_eval)
        plt.title('Mitigation Efficiency')
        plt.ylabel('Efficiency Score')
        
        plt.tight_layout()
        plt.savefig("evaluation_plot.png")
        print("Evaluation plot saved.")

if __name__ == "__main__":
    plot_results()
