# README.md

**Project:** Simulating Infectious Disease Spread in a Social Network (Enhanced & Fully Fleshed Out)

This repository provides a complete, ready-to-run pipeline:
- Synthetic network generation (Erdős–Rényi, Watts–Strogatz, Barabási–Albert)
- CLI for all tasks (generate, visualize, simulate SI/SIR, parameter sweep, analyze)
- Docker support
- Unit tests
- LaTeX report & Beamer slides

---

## 📂 Directory Structure
```
project_root/
├─ data/                       # Generated networks
│  ├─ network_erdos_renyi.pkl
│  ├─ network_watts_strogatz.pkl
│  └─ network_barabasi_albert.pkl
├─ notebooks/                  # Exploration notebook
│  └─ exploration.ipynb        # (placeholder)
├─ src.                       # Source code
│  ├─ cli.py                  # Command-line interface
│  ├─ config.py               # Default config & JSON loader
│  ├─ utils.py                # Helpers
│  ├─ generate.py             # Network generation
│  ├─ visualize.py            # Plotting functions
│  ├─ simulate.py             # SI & SIR models
│  ├─ analyze.py              # Results analysis
│  └─ sweep.py                # Parameter sweeps
├─ tests/                     # Unit tests
│  ├─ test_generate.py
│  └─ test_simulate.py
├─ results/                   # Outputs
│  ├─ figures/
│  └─ logs/
├─ report/                    # LaTeX deliverables
│  ├─ report.tex
│  └─ slides.tex
├─ config.json                # Example configuration
├─ requirements.txt
├─ Dockerfile
└─ README.md                  # This file
```

---

## 🔧 requirements.txt
```text
networkx>=2.8
matplotlib>=3.5
numpy>=1.22
pandas>=1.4
click>=8.1
pytest>=7.0
```

---

## 🐳 Dockerfile
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENTRYPOINT ["python", "src.cli.py"]
``` 

---

## ⚙️ config.json
```json
{
  "model": "erdos_renyi",           // Options: erdos_renyi, watts_strogatz, barabasi_albert, stochastic_block
  "N": 5000,
  "p_edge": 0.001,
  "k_ws": 6,
  "p_ws": 0.1,
  "m_ba": 3,
  "sbm_sizes": [2000, 1500, 1500],        // sizes of communities for stochastic_block
  "sbm_probs": [                        // community connection probabilities
    [0.8, 0.05, 0.05],
    [0.05, 0.7, 0.1],
    [0.05, 0.1, 0.6]
  ],
  "initial_infected_frac": 0.01,
  "p_transmission": 0.03,
  "time_steps": 50,
  "recovery_prob": 0.0,
  "seed": 42
}
```

---

## 🔨 src.config.py
```python
import json
from pathlib import Path

# Default parameters
defaults = {
    "model": "erdos_renyi",
    "N": 5000,
    "p_edge": 0.001,
    "k_ws": 6,
    "p_ws": 0.1,
    "m_ba": 3,
    "initial_infected_frac": 0.01,
    "p_transmission": 0.03,
    "time_steps": 50,
    "recovery_prob": 0.0,
    "seed": 42
}

# Load config.json if exists
def load_config(path: str = 'config.json') -> dict:
    cfg = defaults.copy()
    p = Path(path)
    if p.exists():
        with open(p) as f:
            user_cfg = json.load(f)
        cfg.update(user_cfg)
    return cfg

config = load_config()
``` 

---

## 🔨 src.utils.py
```python
import pickle
from pathlib import Path

def save_pickle(obj, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def load_pickle(path: str):
    with open(path, 'rb') as f:
        return pickle.load(f)
``` 

---

## 🔨 src.generate.py
```python
import networkx as nx
from src.config import config
from src.utils import save_pickle


def generate_network():
    N = config['N']
    model = config['model']
    if model == 'erdos_renyi':
        G = nx.erdos_renyi_graph(N, config['p_edge'], seed=config['seed'])
    elif model == 'watts_strogatz':
        G = nx.watts_strogatz_graph(N, config['k_ws'], config['p_ws'], seed=config['seed'])
    elif model == 'barabasi_albert':
        G = nx.barabasi_albert_graph(N, config['m_ba'], seed=config['seed'])
    elif model == 'stochastic_block':
        # Stochastic Block Model for community structure
        sizes = config.get('sbm_sizes')
        probs = config.get('sbm_probs')
        G = nx.stochastic_block_model(sizes, probs, seed=config['seed'])
    else:
        raise ValueError(f"Unknown model: {model}")
    save_pickle(G, f"data/network_{model}.pkl")
    print(f"Saved network ({model}): {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")


if __name__ == '__main__':
    generate_network()
```

---

## 🔨 src.visualize.py
```python
import networkx as nx
import matplotlib.pyplot as plt
from src.config import config
from src.utils import load_pickle

def plot_network(graph_path: str, infected=None, out_path: str = 'results/figures/network.png'):
    G = load_pickle(graph_path)
    pos = nx.spring_layout(G, seed=config['seed'])
    plt.figure(figsize=(8,8))
    if infected:
        healthy = set(G.nodes()) - set(infected)
        nx.draw_networkx_nodes(G, pos, nodelist=list(healthy), node_color='blue', node_size=10)
        nx.draw_networkx_nodes(G, pos, nodelist=list(infected), node_color='red', node_size=10)
    else:
        nx.draw(G, pos, node_size=10, edge_color='gray', alpha=0.3)
    plt.axis('off')
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved visualization: {out_path}")

if __name__ == '__main__':
    model = config['model']
    plot_network(f"data/network_{model}.pkl")
``` 

---

## 🔨 src.simulate.py
```python
import random
from src.config import config
from src.utils import load_pickle, save_pickle


def initialize_infected(G):
    random.seed(config['seed'])
    k = max(1, int(config['initial_infected_frac'] * G.number_of_nodes()))
    return set(random.sample(list(G.nodes()), k))


def run_SI(G):
    infected = initialize_infected(G)
    history = [len(infected)]
    for _ in range(config['time_steps']):
        new = set()
        for u in infected:
            for v in G.neighbors(u):
                if v not in infected and random.random() < config['p_transmission']:
                    new.add(v)
        infected |= new
        history.append(len(infected))
    return history, infected


def run_SIR(G):
    infected = initialize_infected(G)
    recovered = set()
    history = [(len(infected), len(recovered))]
    for _ in range(config['time_steps']):
        new_inf, new_rec = set(), set()
        for u in infected:
            if random.random() < config['recovery_prob']:
                new_rec.add(u)
            else:
                for v in G.neighbors(u):
                    if v not in infected and v not in recovered and random.random() < config['p_transmission']:
                        new_inf.add(v)
        infected |= new_inf
        infected -= new_rec
        recovered |= new_rec
        history.append((len(infected), len(recovered)))
    return history, (infected, recovered)

if __name__ == '__main__':
    G = load_pickle(f"data/network_{config['model']}.pkl")
    if 'SIR' in config and config['recovery_prob'] > 0:
        history, final = run_SIR(G)
        save_pickle(history, f"results/logs/history_SIR.pkl")
        print("SIR simulation complete.")
    else:
        history, final = run_SI(G)
        save_pickle(history, f"results/logs/history_SI.pkl")
        print("SI simulation complete.")
``` 

---

## 🔨 src.analyze.py
```python
import matplotlib.pyplot as plt
import pandas as pd
from src.utils import load_pickle
from src.config import config

def plot_SI(history, out_path='results/figures/infection_curve_si.png'):
    plt.plot(history)
    plt.xlabel('Time step')
    plt.ylabel('# infected')
    plt.title('SI Model')
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved SI curve: {out_path}")


def plot_SIR(history, out_path='results/figures/infection_curve_sir.png'):
    inf, rec = zip(*history)
    plt.plot(inf, label='Infected')
    plt.plot(rec, label='Recovered')
    plt.xlabel('Time step')
    plt.ylabel('Count')
    plt.legend()
    plt.title('SIR Model')
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved SIR curve: {out_path}")

if __name__ == '__main__':
    if config['recovery_prob'] > 0:
        hist = load_pickle('results/logs/history_SIR.pkl')
        plot_SIR(hist)
    else:
        hist = load_pickle('results/logs/history_SI.pkl')
        plot_SI(hist)
``` 

---

## 🔨 src.sweep.py
```python
import numpy as np
import pandas as pd
from src.simulate import run_SI
from src.utils import load_pickle, save_pickle
from src.config import config

def parameter_sweep(param: str, values: list):
    results = []
    G = load_pickle(f"data/network_{config['model']}.pkl")
    for v in values:
        config[param] = v
        history, _ = run_SI(G)
        results.append({'param': param, 'value': v, 'final_infected': history[-1]})
    df = pd.DataFrame(results)
    save_pickle(df, 'results/logs/sweep_summary.pkl')
    df.to_csv('results/logs/sweep_summary.csv', index=False)
    print("Sweep complete. Summary saved.")

if __name__ == '__main__':
    import sys
    _, param, *vals = sys.argv
    values = list(map(float, vals))
    parameter_sweep(param, values)
``` 

---

## 🔨 src.cli.py
```python
import click
from src.generate import generate_network
from src.visualize import plot_network
from src.simulate import run_SI, run_SIR
from src.analyze import plot_SI, plot_SIR
from src.sweep import parameter_sweep
from src.config import config

@click.group()
def cli():
    """Disease Spread Simulation CLI"""
    pass

@cli.command()
@click.option('--model', default=None, help='Network model: erdos_renyi, watts_strogatz, barabasi_albert, stochastic_block')
def generate(model):
    if model: config['model'] = model
    generate_network()

@cli.command()
@click.option('--input', 'inp', default=None, help='Path to graph pickle')
def visualize(inp):
    path = inp or f"data/network_{config['model']}.pkl"
    plot_network(path)

@cli.command()
@click.option('--model_type', default='SI', help='SI or SIR')
def simulate(model_type):
    from src.utils import load_pickle
    G = load_pickle(f"data/network_{config['model']}.pkl")
    if model_type.upper() == 'SIR' and config['recovery_prob']>0:
        history, _ = run_SIR(G)
        plot_SIR(history)
    else:
        history, _ = run_SI(G)
        plot_SI(history)

@cli.command()
@click.option('--param', required=True, help='Parameter name (e.g., p_transmission)')
@click.option('--values', multiple=True, type=float, help='Values to sweep')
def sweep(param, values):
    parameter_sweep(param, list(values))

@cli.command()
@click.option('--history', default=None, help='Path to history pickle')
def analyze(history):
    from src.utils import load_pickle
    path = history or ('results/logs/history_SIR.pkl' if config['recovery_prob']>0 else 'results/logs/history_SI.pkl')
    try:
        hist = load_pickle(path)
    except FileNotFoundError:
        fallback = 'results/logs/history_SI.pkl'
        hist = load_pickle(fallback)
        print(f"Warning: {{path}} not found. Falling back to SI history {{fallback}}.")
    if isinstance(hist[0], tuple):
        plot_SIR(hist)
    else:
        plot_SI(hist)

if __name__ == '__main__':
    cli()
```

---

## 🧪 tests/test_generate.py
```python
import os
from src.generate import generate_network

def test_generate_creates_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    # override config for small N
    import src.config as c; c.config['N']=10
    generate_network()
    assert (tmp_path / 'data' / 'network_erdos_renyi.pkl').exists()
``` 

---

## 🧪 tests/test_simulate.py
```python
import networkx as nx
from src.simulate import run_SI

def test_run_SI_small_graph():
    G = nx.path_graph(5)
    history, infected = run_SI(G)
    assert isinstance(history, list)
    assert history[0] >= 1
``` 

---

## 📄 report/report.tex
```latex
\documentclass{article}
\usepackage{graphicx}
\title{Simulating Infectious Disease Spread in a Social Network}
\author{Your Name}
\date{\today}
\begin{document}
\maketitle
\section{Introduction}
We model disease spread using SI and SIR on a synthetic social network of $N=5000$.
\section{Methods}
\subsection{Network Generation}
We use Erdős–Rényi, Watts–Strogatz, and Barabási–Albert models (NetworkX).
\subsection{Simulation}
SI: infected remain infectious forever.\\
SIR: infected recover with probability $r$ per time step.
\section{Results}
\includegraphics[width=0.8\linewidth]{../results/figures/network.png}
\includegraphics[width=0.8\linewidth]{../results/figures/infection_curve_si.png}
\section{Discussion}
Assumptions: homogeneous mixing, no births/deaths.
\section{Conclusion}
Pipeline is modular and extensible.
\end{document}
``` 

---

## 📄 report/slides.tex
```latex
\documentclass{beamer}
\usetheme{Madrid}
\title{Disease Spread Simulation}
\author{Your Name}
\date{\today}
\begin{document}
\frame{\titlepage}
\begin{frame}{Network}
\includegraphics[width=0.7\linewidth]{../results/figures/network.png}
\end{frame}
\begin{frame}{SI Model}
\includegraphics[width=0.7\linewidth]{../results/figures/infection_curve_si.png}
\end{frame}
\begin{frame}{Next Steps}
\begin{itemize}
  \item Add mobility
  \item Calibrate to real data
  \item Expand to SEIR
\end{itemize}
\end{frame}
\end{document}
``` 

---

### 🚀 To Run Everything
1. **Install**: `pip install -r requirements.txt`
2. **Generate**: `python src.cli.py generate`
3. **Visualize**: `python src.cli.py visualize`
4. **Simulate SI**: `python src.cli.py simulate --model_type SI`
5. **Analyze SI**: `python src.cli.py analyze`
6. **Simulate SIR**: update `recovery_prob` in `config.json`, then `python src.cli.py simulate --model_type SIR`
7. **Sweep**: `python src.cli.py sweep --param p_transmission --values 0.01 0.03 0.05`
8. **Run tests**: `pytest`
9. **Report/Slides**: `pdflatex -output-directory report report/report.tex` & `slides.tex`

python -m src.cli generate
python -m src.cli visualize
python -m src.cli metrics
python -m src.cli simulate --model SI
python -m src.cli simulate --model SIR
python -m src.cli analyze
python -m src.cli sweep --param p_transmission --values 0.01 --values 0.03 --values 0.05 --values 0.10
pytest
dlatex report/report.tex && pdflatex report/slides.tex
