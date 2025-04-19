import pandas as pd
from src.config import config
from src.utils import load_pickle, save_pickle
from src.simulate import run_SI
from pathlib import Path

def parameter_sweep(param: str, values: list):
    G = load_pickle(f"data/network_{config['model']}.pkl")
    results = []
    for v in values:
        config[param] = v
        history, _ = run_SI(G)
        results.append({'value': v, 'final': history[-1]})
    df = pd.DataFrame(results)
    Path('results/logs').mkdir(parents=True, exist_ok=True)
    df.to_csv('results/logs/sweep.csv', index=False)
    print("[sweep] done, results in results/logs/sweep.csv")

if __name__ == '__main__':
    import sys
    _, param, *vals = sys.argv
    parameter_sweep(param, list(map(float, vals)))