# Sentinel: Neuro-Symbolic DDoS Defense Framework

This repository contains the source code for the paper **"Sentinel: A Neuro-Symbolic Co-Evolutionary Framework for Autonomous DDoS Defense"**, submitted to *IEEE Transactions on Network and Service Management*.

## Overview
Sentinel is an autonomous defense agent that combines **Deep Reinforcement Learning (DRL)** with a **Symbolic Safety Layer**. It is trained using a **Co-Evolutionary Genetic Algorithm** with a **Hall of Fame (HoF)** mechanism to ensure robustness against polymorphic attacks.

## Key Features
-   **Neuro-Symbolic Architecture**: Combines neural pattern matching with logical safety constraints.
-   **Co-Evolutionary Training**: Attacker and Defender agents evolve in an arms race.
-   **Hall of Fame**: Prevents catastrophic forgetting by training against historical champions.
-   **Zero-Shot Generalization**: Proven effectiveness against unseen attack vectors (e.g., ICMP Floods).
-   **Explainable AI (XAI)**: Generates human-readable explanations for defense actions.

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/AliAlfatemi/sentinel-ddos.git
    cd sentinel-ddos
    ```

2.  Create a virtual environment:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### 1. Training (Massive Scale)
To reproduce the 2000-generation longitudinal study:
```bash
python training/train.py
```
*Note: This will take several hours. Results will be saved to `training_results_hof.csv`.*

### 2. Evaluation
To run the comparative evaluation against baselines:
```bash
python analysis/evaluate_system.py
```

### 3. Advanced Analysis (Heatmaps & Adaptation)
To generate the behavioral analysis figures (Heatmap, Adaptation Speed):
```bash
python analysis/advanced_evaluation.py
```

### 4. Safety Analysis (Chaos Mode)
To test the safety layer in chaotic conditions:
```bash
python analysis/safety_analysis.py
```

## Project Structure
-   `agents/`: Source code for DRL, Neuro-Symbolic, and Baseline agents.
-   `simulation/`: Custom Gymnasium environment (`DDoSEnv`).
-   `training/`: Training loops and genetic algorithm logic.
-   `analysis/`: Scripts for evaluation, visualization, and XAI.

## Citation
If you use this code, please cite our paper:
```bibtex
@article{alfatemi2025sentinel,
  title={Sentinel: A Neuro-Symbolic Co-Evolutionary Framework for Autonomous DDoS Defense},
  author={Ali Alfatemi, Mohamed Rahouti, Zakirul Alam Bhuiyan, Ahmed Alfaqeer},
  journal={IEEE Transactions on Network and Service Management},
  year={2025}
}
```
