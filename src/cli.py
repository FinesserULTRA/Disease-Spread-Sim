import click
from src.generate import generate_network
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
def generate():
    "Generate the synthetic network"
    generate_network()

@cli.command()
def visualize():
    "Plot the network structure"
    plot_network(f"data/network_{config['model']}.pkl")

@cli.command()
def metrics():
    "Compute degree distribution & centrality"
    G = load_pickle(f"data/network_{config['model']}.pkl")
    compute_degree_distribution(G)
    compute_centrality(G)

@cli.command()
@click.option('--model', default='SI', help='SI or SIR')
def simulate(model):
    "Run SI or SIR simulation"
    G = load_pickle(f"data/network_{config['model']}.pkl")
    if model.upper()=='SIR' and config['recovery_prob']>0:
        history, _ = run_SIR(G)
        save_path='results/logs/history_SIR.pkl'
    else:
        history, _ = run_SI(G)
        save_path='results/logs/history_SI.pkl'
    save_pickle(history, save_path)
    click.echo(f"Simulation saved to {save_path}")

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
        print(f"[analyze] fallback to SI history {fallback}")
    if isinstance(hist[0], tuple):
        plot_SIR(hist)
    else:
        plot_SI(hist)

@cli.command()
@click.option('--param', required=True)
@click.option('--values', required=True, multiple=True, type=float)
def sweep(param, values):
    "Parameter sweep for given values"
    parameter_sweep(param, list(values))

if __name__=='__main__':
    cli()