# Integrated Disease Spread Simulation
# Complete Flow from Network Creation → Visualization → Simulation → Analysis

# --------------------------- config.json ---------------------------
# Configuration file for simulation settings

{
  "model": "stochastic_block",
  "N": 5000,
  "p_edge": 0.001,
  "k_ws": 6,
  "p_ws": 0.1,
  "m_ba": 3,
  "sbm_sizes": [2000, 1500, 1500],
  "sbm_probs": [ [0.8,0.05,0.05], [0.05,0.7,0.1], [0.05,0.1,0.6] ],
  "initial_infected_frac": 0.01,
  "p_transmission": 0.03,
  "recovery_prob": 0.0,
  "time_steps": 50,
  "seed": 42
}

# --------------------------- config.py ---------------------------
import json
from pathlib import Path

def load_config(path='config.json'):
    default = {
        "model": "erdos_renyi",
        "N": 500,
        "p_edge": 0.01,
        "k_ws": 4,
        "p_ws": 0.1,
        "m_ba": 3,
        "sbm_sizes": [200, 150, 150],
        "sbm_probs": [[0.8, 0.05, 0.05], [0.05, 0.7, 0.1], [0.05, 0.1, 0.6]],
        "initial_infected_frac": 0.02,
        "p_transmission": 0.04,
        "recovery_prob": 0.0,
        "time_steps": 40,
        "seed": 42
    }
    cfg = default.copy()
    if Path(path).exists():
        cfg.update(json.loads(Path(path).read_text()))
    return cfg

config = load_config()

# --------------------------- utils.py ---------------------------
import pickle
from pathlib import Path

def save_pickle(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

# --------------------------- generate.py ---------------------------
import networkx as nx
import random
from config import config
from utils import save_pickle

def generate_network():
    model = config['model']
    N = config['N']
    seed = config['seed']
    if model == 'erdos_renyi':
        G = nx.erdos_renyi_graph(N, config['p_edge'], seed=seed)
    elif model == 'watts_strogatz':
        G = nx.watts_strogatz_graph(N, config['k_ws'], config['p_ws'], seed=seed)
    elif model == 'barabasi_albert':
        G = nx.barabasi_albert_graph(N, config['m_ba'], seed=seed)
    elif model == 'stochastic_block':
        G = nx.stochastic_block_model(config['sbm_sizes'], config['sbm_probs'], seed=seed)
    else:
        raise ValueError("Unknown model")

    save_pickle(G, f'data/network_{model}.pkl')
    print(f"Generated {model} network with {G.number_of_nodes()} nodes.")
    return G

# --------------------------- visualize.py ---------------------------
import matplotlib.pyplot as plt
import networkx as nx
from utils import load_pickle

def plot_network(path, infected=None):
    G = load_pickle(path)
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(8, 8))
    if infected:
        healthy = list(set(G.nodes()) - set(infected))
        nx.draw_networkx_nodes(G, pos, nodelist=healthy, node_color='skyblue', node_size=10)
        nx.draw_networkx_nodes(G, pos, nodelist=list(infected), node_color='red', node_size=10)
    else:
        nx.draw_networkx_nodes(G, pos, node_size=10, node_color='lightgray')
    nx.draw_networkx_edges(G, pos, alpha=0.2)
    plt.title("Social Network")
    plt.axis('off')
    plt.savefig("results/figures/network_plot.png")
    plt.close()

# --------------------------- simulate.py ---------------------------
import random
from utils import load_pickle, save_pickle
from config import config

def simulate_spread(path):
    G = load_pickle(path)
    infected = set(random.sample(list(G.nodes()), int(config['initial_infected_frac'] * G.number_of_nodes())))
    history = [len(infected)]

    for t in range(config['time_steps']):
        new_infected = set()
        for node in infected:
            for neighbor in G.neighbors(node):
                if neighbor not in infected and random.random() < config['p_transmission']:
                    new_infected.add(neighbor)
        infected |= new_infected
        history.append(len(infected))

    save_pickle(history, "results/logs/infection_history.pkl")
    return history

# --------------------------- analyze.py ---------------------------
import matplotlib.pyplot as plt
import pandas as pd
from utils import load_pickle

def plot_results():
    hist = load_pickle("results/logs/infection_history.pkl")
    df = pd.DataFrame({"Time": range(len(hist)), "Infected": hist})
    df.to_csv("results/logs/infection_summary.csv", index=False)
    plt.figure()
    plt.plot(df["Time"], df["Infected"], label="Infected", color="red")
    plt.xlabel("Time")
    plt.ylabel("Number of Infected Individuals")
    plt.title("Infection Spread Over Time")
    plt.legend()
    plt.grid(True)
    plt.savefig("results/figures/infection_curve.png")
    plt.close()

# --------------------------- run.py ---------------------------
from generate import generate_network
from visualize import plot_network
from simulate import simulate_spread
from analyze import plot_results
from config import config

if __name__ == '__main__':
    G = generate_network()
    plot_network(f"data/network_{config['model']}.pkl")
    simulate_spread(f"data/network_{config['model']}.pkl")
    plot_results()
    print("Simulation complete. Results saved in 'results/' directory.")
