import networkx as nx
from src.simulate import run_SI

def test_run_SI_small_graph():
    G = nx.path_graph(5)
    history, infected = run_SI(G)
    assert isinstance(history, list)
    assert history[0] >= 1