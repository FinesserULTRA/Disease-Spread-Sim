import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from src.utils import save_pickle, load_pickle
from src.config import config


def compute_degree_distribution(G, out_csv='results/metrics/degree.csv', out_png='results/figures/degree_hist.png'):
    degs = dict(G.degree())
    df = pd.DataFrame({'node': list(degs), 'degree': list(degs.values())})
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    plt.figure()
    df['degree'].hist(bins=50)
    plt.xlabel('Degree'); plt.ylabel('Frequency'); plt.title('Degree Distribution')
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"[metrics] degree CSV + histogram saved")


def compute_centrality(G, out_csv='results/metrics/centrality.csv'):
    deg = nx.degree_centrality(G)
    btw = nx.betweenness_centrality(G)
    clus = nx.clustering(G)
    df = pd.DataFrame({
        'node': list(G.nodes()),
        'degree_centrality': [deg[n] for n in G.nodes()],
        'betweenness': [btw[n] for n in G.nodes()],
        'clustering': [clus[n] for n in G.nodes()]
    })
    df.to_csv(out_csv, index=False)
    print(f"[metrics] centrality CSV saved")