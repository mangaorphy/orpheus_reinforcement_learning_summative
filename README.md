# 🦁 Savanna Acoustic Threat Detection — RL Summative

> **Mission**: Develop an AI-based acoustic monitoring system that analyzes wildlife vocalizations to detect changes in spatial and behavioral patterns, enabling early identification of ecological threats and supporting data-driven conservation in African ecosystems.

---

## Project Overview

An AI ranger drone navigates a 20×20 African savanna grid, monitoring wildlife acoustic signals to identify ecological threats (poaching, predator proximity, habitat disturbance). Four reinforcement learning algorithms (DQN, REINFORCE, PPO, A2C) are trained and compared on this mission-critical task.

---

## Repository Structure

```
savanna_rl/
├── environment/
│   ├── custom_env.py        # Custom Gymnasium environment
│   └── rendering.py         # Pygame visualization + random demo
├── training/
│   ├── dqn_training.py      # DQN — 10 hyperparameter runs
│   └── pg_training.py       # REINFORCE / PPO / A2C — 10 runs each
├── models/
│   ├── dqn/                 # Saved DQN models
│   └── pg/                  # Saved policy gradient models
├── results/                 # JSON results per run
├── logs/                    # TensorBoard logs
├── main.py                  # Entry point — run best model or demo
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the random agent demo (no training required)
```bash
python main.py --demo
```
This opens the Pygame GUI showing the drone taking **random actions** — no model involved.

### 3. Train all algorithms
```bash
# Train DQN (10 hyperparameter runs)
python training/dqn_training.py

# Train policy gradient methods (REINFORCE, PPO, A2C)
python training/pg_training.py

# Train only best configs (faster)
python training/dqn_training.py --run best
python training/pg_training.py --algo ppo --run best
```

### 4. Run the best-performing agent
```bash
python main.py --algo ppo --episodes 3
python main.py --algo dqn --episodes 3
python main.py --compare      # compare all saved models
```

---

## Environment Details

### Observation Space
Continuous box, shape **(847,)** — float32:
| Component | Dim | Description |
|---|---|---|
| Drone position (x, y) | 2 | Normalized [0,1] |
| Battery fraction | 1 | Current battery / 100 |
| Step fraction | 1 | Current step / 500 |
| Acoustic heatmap | 400 | 20×20 flattened signal intensity |
| Confirmed threat map | 400 | 20×20 confirmed threat locations |
| Local species calls | 36 | 4 species × 9 neighbourhood cells |
| Last action (one-hot) | 7 | Previous action encoding |

### Action Space
**Discrete(7)**:
| ID | Action | Battery Cost |
|---|---|---|
| 0 | Move North | 2 |
| 1 | Move South | 2 |
| 2 | Move East | 2 |
| 3 | Move West | 2 |
| 4 | **Scan** (deep acoustic analysis) | 5 |
| 5 | **Alert** (radio to ranger base) | 1 |
| 6 | Hover (passive listening) | 0.5 |

### Reward Structure
| Event | Reward |
|---|---|
| Confirm new threat via SCAN | +10 |
| Flagship species detected (elephant/lion) | +5 |
| Alert confirmed+scanned threat | +8 per threat |
| Visit new cell (coverage) | +2 |
| Approach acoustic hotspot | +signal × 1.0 |
| Hover at hotspot | +signal × 0.5 |
| False alarm (alert without scan) | −3 |
| Wasted scan (nothing found) | −2 |
| Revisit recent cell | −1 |
| Distant unresolved poaching | −0.2/step |
| Step cost (efficiency) | −0.5 |
| Battery depleted | −20 |
| Unresolved threats at episode end | −5/threat |
| **Mission complete** (all threats alerted) | **+50** |

### Terminal Conditions
1. Battery depleted (≤ 0)
2. Max steps reached (500)
3. All 5 threats confirmed + alerted (**mission complete**)

### Biomes
- **Open savanna** — default terrain
- **Woodland** — attenuates acoustic signal (×0.6)
- **Waterhole** — amplifies signal (×1.3), wildlife magnet
- **Rocky outcrop** — lion territory

---

## RL Algorithms

### DQN (Value-Based)
Uses experience replay and target network. Key insight: acoustic hotspots provide natural reward shaping that helps DQN's ε-greedy exploration.

**Key hyperparameters tuned** (10 runs):
- `learning_rate` — {1e-4, 5e-4, 2e-4}
- `buffer_size` — {50k, 100k, 200k}
- `exploration_fraction` — {0.15, 0.2, 0.3, 0.5}
- `gamma` — {0.90, 0.99, 0.995}
- `tau` (soft update) — {0.01, 0.05, 1.0}
- `net_arch` — {[128,128], [256,256], [256,128,64], [256,256,128]}

### REINFORCE (Policy Gradient)
Monte-Carlo returns with optional baseline. Implemented from scratch using PyTorch for full transparency.

**Key hyperparameters tuned** (10 runs):
- `learning_rate` — {1e-4, 1e-3, 2e-3, 5e-3}
- `gamma` — {0.90, 0.99, 0.995}
- `use_baseline` — {True, False}
- `entropy_coef` — {0.0, 0.01, 0.02, 0.05}
- `net_arch` — {(128,128), (256,128), (256,256,128), (512,256)}

### PPO (Proximal Policy Optimization)
Clipped surrogate objective prevents destructive policy updates. Well-suited for the continuous acoustic observation space.

**Key hyperparameters tuned** (10 runs):
- `n_steps` — {512, 2048, 4096}
- `clip_range` — {0.1, 0.2, 0.4}
- `n_epochs` — {10, 15, 20}
- `gae_lambda` — {0.80, 0.95}
- `ent_coef` — {0.0, 0.01}
- `gamma` — {0.99, 0.995}

### A2C (Advantage Actor-Critic)
Synchronous advantage estimation. Faster per-step than PPO with fewer hyperparameters.

**Key hyperparameters tuned** (10 runs):
- `n_steps` — {5, 10, 20}
- `gae_lambda` — {0.95, 1.0}
- `ent_coef` — {0.0, 0.01}
- `vf_coef` — {0.5, 1.0}
- `max_grad_norm` — {0.1, 0.5}
- `gamma` — {0.90, 0.99, 0.995}

---

## Visualization

The Pygame renderer displays:
- Colour-coded biome grid (savanna/woodland/waterhole/rocky)
- Acoustic intensity heatmap (green→red gradient)
- Animated threat pulse rings (unresolved = pulsing, confirmed = yellow)
- Species call halos (pulsing per call frequency)
- Drone with battery arc indicator
- Real-time HUD: step, battery bar, threat status, coverage %

---

## Author & Course Info

**Assignment**: RL Summative — Compare Value-Based vs Policy Gradient Methods  
**Environment**: African Savanna Acoustic Threat Detection (Custom Gymnasium)  
**Algorithms**: DQN · REINFORCE · PPO · A2C  
**Library**: Stable-Baselines3 v2.x  
**Python**: 3.9+


pip install -r requirements.txt
python main.py --demo                        # random agent viz (Task 2a)
python training/dqn_training.py --run best  # quick best-config run
python training/pg_training.py --algo ppo --run best
python main.py --compare                     # compare all algorithms