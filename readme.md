# Project: Simulating Infectious Disease Spread in a Social Network

A turnkey pipeline for generating synthetic or ingesting real-world social networks, visualizing them, simulating SI/SIR disease spread, analyzing results, and producing deliverables.

---

## Directory Structure
```
project_root/
├─ README.md                  # Project overview and quickstart
├─ requirements.txt           # Python dependencies
├─ config.json                # User-configurable parameters
├─ Dockerfile                 # Container definition
├─ data/                      # Saved NetworkX graphs
│  ├─ network_erdos_renyi.pkl
│  ├─ network_watts_strogatz.pkl
│  ├─ network_barabasi_albert.pkl
│  └─ network_stochastic_block.pkl
├─ src/                       # Source modules
│  ├─ config.py               # Load defaults & user overrides
│  ├─ utils.py                # I/O helpers
│  ├─ generate.py             # Synthetic generation & ingestion
│  ├─ visualize.py            # Force-directed plotting
│  ├─ metrics.py              # Degree & centrality analysis
│  ├─ simulate.py             # SI & SIR simulation
│  ├─ analyze.py              # Infection curve plotting & CSV summary
│  ├─ sweep.py                # Parameter sweep experiments
│  └─ cli.py                  # Click-based command-line interface
├─ results/                   # Outputs
│  ├─ figures/                # PNG plots
│  └─ logs/                   # Pickled histories, CSV summaries
├─ tests/                     # PyTest tests
│  ├─ test_generate.py
│  └─ test_simulate.py
├─ notebooks/                 # Jupyter exploration notebooks
│  └─ exploration.ipynb
└─ report/                    # LaTeX deliverables
   ├─ report.tex              # Detailed write-up
   └─ slides.tex              # Beamer presentation
```

---

## README.md

# Simulating Infectious Disease Spread in a Social Network

Features:
- Synthetic network models: ER, WS, BA, SBM
- Real-world network ingestion (edge lists or CSV)
- CLI & optional Docker container
- Visualization: force-directed layout, degree histograms, centrality heatmaps
- Epidemic models: SI and SIR, with parameter sweeping
- Metrics: infection curves, peak stats, network metrics (degree, centrality)
- Unit tests and Jupyter notebooks for exploratory analysis
- LaTeX report and Beamer slides

## Quickstart

```bash
# Clone and install dependencies
git clone <repo_url> && cd <repo_dir>
python3 -m venv env; source env/bin/activate
pip install -r requirements.txt

# (Optional) Docker build
docker build -t disease-sim .
```

### Examples
```bash
# Ingest real-world network
python -m src.cli ingest --input path/to/edgelist.csv
# Generate synthetic network
python -m src.cli generate --model erdos_renyi
# Visualize and compute metrics
python -m src.cli visualize --input data/network_real.pkl
python -m src.cli metrics --input data/network_real.pkl
# Simulate SI or SIR
python -m src.cli simulate --model SI
python -m src.cli simulate --model SIR
# Analyze results
python -m src.cli analyze
# Parameter sweep
python -m src.cli sweep --param p_transmission --values 0.01 0.03 0.05
# Run tests
pytest
# Compile reports
pdflatex report/report.tex
pdflatex report/slides.tex
```

---

## requirements.txt
```text
networkx>=2.8
matplotlib>=3.5
numpy>=1.22
pandas>=1.4
click>=8.1
pytest>=7.0
``` 

---

## Dockerfile
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . ./
ENTRYPOINT ["python", "-m", "src.cli"]
``` 

---

## config.json
```json
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
``` 

---

# src/config.py
```python
import json
from pathlib import Path

def load_config(path='config.json') -> dict:
    defaults = {
        'model': 'stochastic_block', 'N': 5000, 'p_edge': 0.001,
        'k_ws': 6, 'p_ws': 0.1, 'm_ba': 3,
        'sbm_sizes': [2000,1500,1500],
        'sbm_probs': [[0.8,0.05,0.05],[0.05,0.7,0.1],[0.05,0.1,0.6]],
        'initial_infected_frac': 0.01, 'p_transmission':0.03,
        'recovery_prob':0.0, 'time_steps':50, 'seed':42
    }
    cfg = defaults.copy()
    if Path(path).exists():
        cfg.update(json.loads(Path(path).read_text()))
    return cfg

config = load_config()
``` 

---

# src/utils.py
```python
import pickle
from pathlib import Path

def save_pickle(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path,'wb') as f: pickle.dump(obj,f)

def load_pickle(path):
    with open(path,'rb') as f: return pickle.load(f)
``` 

---

# src/generate.py
```python
import networkx as nx
from src.config import config
from src.utils import save_pickle

def load_real_network(input_path):
    try:
        G = nx.read_edgelist(input_path)
    except:
        import pandas as pd
        df = pd.read_csv(input_path)
        G = nx.from_pandas_edgelist(df, df.columns[0], df.columns[1])
    save_pickle(G,'data/network_real.pkl')
    print(f"Loaded real network: {G.number_of_nodes()} nodes")
    return G


def generate_network():
    N, model, s = config['N'], config['model'], config['seed']
    if model=='erdos_renyi': G=nx.erdos_renyi_graph(N,config['p_edge'],seed=s)
    elif model=='watts_strogatz': G=nx.watts_strogatz_graph(N,config['k_ws'],config['p_ws'],seed=s)
    elif model=='barabasi_albert': G=nx.barabasi_albert_graph(N,config['m_ba'],seed=s)
    elif model=='stochastic_block': G=nx.stochastic_block_model(config['sbm_sizes'],config['sbm_probs'],seed=s)
    else: raise ValueError(model)
    save_pickle(G,f"data/network_{model}.pkl")
    print(f"Generated {model}: {G.number_of_nodes()} nodes")

if __name__=='__main__': generate_network()
``` 

---

# src/visualize.py
```python
import networkx as nx
import matplotlib.pyplot as plt
from src.config import config
from src.utils import load_pickle

def plot_network(path, infected=None, out='results/figures/network.png'):
    G=load_pickle(path)
    pos=nx.spring_layout(G,seed=config['seed'])
    plt.figure(figsize=(8,8))
    if infected:
        h=set(G.nodes())-set(infected)
        nx.draw_networkx_nodes(G,pos,nodelist=list(h),node_color='blue',node_size=10)
        nx.draw_networkx_nodes(G,pos,nodelist=list(infected),node_color='red',node_size=10)
    else:
        nx.draw(G,pos,node_size=10,edge_color='gray',alpha=0.3)
    plt.axis('off'); plt.savefig(out,dpi=300); plt.close()
    print(f"Saved {out}")
``` 

---

# src/metrics.py
```python
import networkx as nx; import pandas as pd; import matplotlib.pyplot as plt
from pathlib import Path
from src.utils import load_pickle

def compute_degree_distribution(G, csv='results/metrics/degree.csv', png='results/figures/degree_hist.png'):
    deg=dict(G.degree()); df=pd.DataFrame(deg.items(),columns=['node','degree'])
    Path(csv).parent.mkdir(parents=True,exist_ok=True); df.to_csv(csv,index=False)
    plt.figure(); df['degree'].hist(bins=50);
    plt.xlabel('Degree');plt.ylabel('Freq'); plt.title('Degree Dist'); plt.savefig(png,dpi=300);plt.close()
    print('Degree histogram saved')

def compute_centrality(G, csv='results/metrics/centrality.csv'):
    deg=nx.degree_centrality(G); btw=nx.betweenness_centrality(G)
    clu=nx.clustering(G); df=pd.DataFrame({'node':list(G.nodes()),
      'deg_cent':[deg[n] for n in G.nodes()],
      'betweenness':[btw[n] for n in G.nodes()],
      'clustering':[clu[n] for n in G.nodes()]})
    Path(csv).parent.mkdir(parents=True,exist_ok=True); df.to_csv(csv,index=False)
    print('Centrality CSV saved')
``` 

---

# src/simulate.py
```python
import random
from src.config import config
from src.utils import load_pickle, save_pickle

def initialize_infected(G):
    random.seed(config['seed'])
    k=max(1,int(config['initial_infected_frac']*G.number_of_nodes()))
    return set(random.sample(list(G.nodes()),k))

def run_SI(G):
    inf=initialize_infected(G); hist=[len(inf)]
    for _ in range(config['time_steps']):
        new=set()
        for u in inf:
            for v in G.neighbors(u):
                if v not in inf and random.random()<config['p_transmission']:
                    new.add(v)
        inf|=new; hist.append(len(inf))
    return hist,inf

def run_SIR(G):
    inf=initialize_infected(G); rec=set(); hist=[(len(inf),len(rec))]
    for _ in range(config['time_steps']):
        newi,newr=set(),set()
        for u in inf:
            if random.random()<config['recovery_prob']: newr.add(u)
            else:
                for v in G.neighbors(u):
                    if v not in inf and v not in rec and random.random()<config['p_transmission']:
                        newi.add(v)
        inf|=newi; inf-=newr; rec|=newr; hist.append((len(inf),len(rec)))
    return hist,(inf,rec)

if __name__=='__main__':
    G=load_pickle(f"data/network_{config['model']}.pkl")
    if config['recovery_prob']>0:
        h,_=run_SIR(G); save_pickle(h,'results/logs/history_SIR.pkl');print('SIR done')
    else:
        h,_=run_SI(G); save_pickle(h,'results/logs/history_SI.pkl');print('SI done')
``` 

---

# src/analyze.py
```python
import matplotlib.pyplot as plt; import pandas as pd
from pathlib import Path
from src.utils import load_pickle
from src.config import config

def plot_SI(hist,out_png='results/figures/curve_si.png',out_csv='results/logs/summary_si.csv'):
    df=pd.DataFrame({'time':range(len(hist)),'infected':hist}); Path(out_csv).parent.mkdir(parents=True,exist_ok=True)
    df.to_csv(out_csv,index=False); plt.figure(); df['infected'].plot(); plt.xlabel('Time');plt.ylabel('Infected');plt.title('SI Curve'); plt.savefig(out_png,dpi=300);plt.close();print('SI plot saved')

def plot_SIR(hist,out_png='results/figures/curve_sir.png',out_csv='results/logs/summary_sir.csv'):
    df=pd.DataFrame(hist,columns=['infected','recovered']); Path(out_csv).parent.mkdir(parents=True,exist_ok=True)
    df.to_csv(out_csv,index=False); plt.figure(); df.plot(); plt.xlabel('Time');plt.title('SIR Curve'); plt.savefig(out_png,dpi=300);plt.close();print('SIR plot saved')

if __name__=='__main__':
    key='history_SIR' if config['recovery_prob']>0 else 'history_SI'
    try: hist=load_pickle(f'results/logs/{key}.pkl')
    except: hist=load_pickle('results/logs/history_SI.pkl');print('Fallback to SI')
    plot_SIR(hist) if isinstance(hist[0],tuple) else plot_SI(hist)
``` 

---

# src/sweep.py
```python
import pandas as pd
from pathlib import Path
from src.config import config
from src.utils import load_pickle
from src.simulate import run_SI

def parameter_sweep(param,values):
    G=load_pickle(f"data/network_{config['model']}.pkl"); results=[]
    for v in values:
        config[param]=v; h,_=run_SI(G); results.append({'value':v,'final':h[-1]})
    df=pd.DataFrame(results); Path('results/logs').mkdir(parents=True,exist_ok=True); df.to_csv('results/logs/sweep.csv',index=False); print('Sweep done')

if __name__=='__main__':
    import sys; _,p,*vals=sys.argv; parameter_sweep(p,list(map(float,vals)))
``` 

---

# src/cli.py
```python
import click
from src.generate import generate_network, load_real_network
from src.visualize import plot_network
from src.metrics import compute_degree_distribution, compute_centrality
from src.simulate import run_SI, run_SIR
from src.analyze import plot_SI, plot_SIR
from src.sweep import parameter_sweep
from src.config import config
from src.utils import load_pickle, save_pickle

@click.group()
def cli(): pass

@cli.command()
@click.option('--input','input_path',default=None,help='Real-world edge list')
def ingest(input_path):
    if not input_path: click.echo('Error'); return
    load_real_network(input_path)

@cli.command()
@click.option('--model',default=None,help='Override model')
def generate(model):
    if model: config['model']=model
    generate_network()

@cli.command()
@click.option('--input','input_path',default=None,help='Graph pickle')
def visualize(input_path):
    plot_network(input_path or f"data/network_{config['model']}.pkl")

@cli.command()
@click.option('--input','input_path',default=None,help='Graph pickle')
def metrics(input_path):
    G=load_pickle(input_path or f"data/network_{config['model']}.pkl"); compute_degree_distribution(G); compute_centrality(G)

@cli.command()
@click.option('--model',default='SI',help='SI or SIR')
@click.option('--input','input_path',default=None,help='Graph pickle')
def simulate(model,input_path):
    G=load_pickle(input_path or f"data/network_{config['model']}.pkl")
    if model.upper()=='SIR' and config['recovery_prob']>0: h,_=run_SIR(G); key='SIR'
    else: h,_=run_SI(G); key='SI'
    save_pickle(h,f'results/logs/history_{key}.pkl'); click.echo(f'Saved history_{key}')

@cli.command()
def analyze():
    plot_SI(None) if config['recovery_prob']==0 else plot_SIR(None)

@cli.command()
@click.option('--param',required=True)
@click.option('--values',required=True,multiple=True,type=float)
def sweep(param,values): parameter_sweep(param,list(values))

if __name__=='__main__': cli()
```

