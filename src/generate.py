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
        G = nx.stochastic_block_model(config['sbm_sizes'], config['sbm_probs'], seed=config['seed'])
    else:
        raise ValueError(f"Unknown model: {model}")
    save_pickle(G, f"data/network_{model}.pkl")
    print(f"[generate] {model}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

if __name__ == '__main__':
    generate_network()