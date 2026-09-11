# Reinforcement Learning for Multi-Criteria Decision Making in a Supply Chain

**Final Year Project**  
**Omole Daniel Oluwatosin**  
Matric Number: 190407050  
Department of Systems Engineering  
Faculty of Engineering  
University of Lagos, Akoka, Lagos, Nigeria  

**Supervisor:** Dr. W. A. Raheem  

---

## Project Overview

This repository contains the complete implementation of a **Reinforcement Learning-based Multi-Criteria Decision Making (RL-MCDM)** framework for dynamic inventory management in a multi-distribution centre (multi-DC) supply chain, calibrated to the Nigerian context.

The system optimises **ordering** and **cross-docking** decisions for a single perishable product — **25kg packaged tomato crates** — under realistic Nigerian conditions including:

- Stochastic and seasonal demand
- Port congestion and fuel price shocks
- Capacity reductions
- Genuine daily spoilage/decay calibrated to Nigerian post-harvest loss data

Two deep reinforcement learning algorithms were implemented and compared:

- **Proximal Policy Optimization (PPO)**
- **Branching Dueling Q-Network (BDQ)**

These were benchmarked against classical inventory policies:

- Economic Order Quantity (EOQ) with safety stock
- Optimised (s, S) continuous-review policy
- Static AHP-TOPSIS Multi-Criteria Decision Making heuristic

---

## Key Results

| Policy              | Mean Cost (₦)     | Fill Rate (%) | vs EOQ      |
|---------------------|-------------------|---------------|-------------|
| **PPO (Best)**      | **802,902,047**   | **78.9%**     | **−3.2%**   |
| EOQ                 | 829,314,778       | 75.8%         | —           |
| (s, S)              | 994,419,324       | 67.9%         | +19.9%      |
| Static MCDM         | 1,736,771,838     | 35.2%         | +109.4%     |
| BDQ                 | 911,631,863       | 75.9%         | +9.9%       |

PPO achieved statistically significant improvements over all classical baselines (p < 0.001).

---

## Repository Structure
