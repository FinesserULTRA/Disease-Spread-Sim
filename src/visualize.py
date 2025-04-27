import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.config import config
from src.utils import load_pickle


def plot_network(path: str, infected=None, out_path="results/figures/network.png"):
    G = load_pickle(path)
    pos = nx.spring_layout(G, seed=config["seed"])
    plt.figure(figsize=(8, 8))
    if infected:
        healthy = set(G.nodes()) - set(infected)
        nx.draw_networkx_nodes(
            G, pos, nodelist=list(healthy), node_color="blue", node_size=10
        )
        nx.draw_networkx_nodes(
            G, pos, nodelist=list(infected), node_color="red", node_size=10
        )
    else:
        nx.draw(G, pos, node_size=10, edge_color="gray", alpha=0.3)
    plt.axis("off")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[visualize] saved {out_path}")


if __name__ == "__main__":
    model = config["model"]
    plot_network(f"data/network_{model}.pkl")
