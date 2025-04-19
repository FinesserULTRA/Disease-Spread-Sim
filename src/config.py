import json
from pathlib import Path

# Default parameters
defaults = {
    "model": "stochastic_block",
    "N": 5000,
    "p_edge": 0.001,
    "k_ws": 6,
    "p_ws": 0.1,
    "m_ba": 3,
    "sbm_sizes": [2000,1500,1500],
    "sbm_probs": [[0.8,0.05,0.05],[0.05,0.7,0.1],[0.05,0.1,0.6]],
    "initial_infected_frac": 0.01,
    "p_transmission": 0.03,
    "recovery_prob": 0.0,
    "time_steps": 50,
    "seed": 42
}

def load_config(path='config.json') -> dict:
    cfg = defaults.copy()
    p = Path(path)
    if p.exists():
        cfg.update(json.loads(p.read_text()))
    return cfg

config = load_config()