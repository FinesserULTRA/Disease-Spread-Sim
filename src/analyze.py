import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from src.utils import load_pickle
from src.config import config


def plot_SI(history, out_png='results/figures/curve_si.png', out_csv='results/logs/summary_si.csv'):
    df = pd.DataFrame({'time': range(len(history)), 'infected': history})
    df.to_csv(out_csv, index=False)
    plt.figure(); df['infected'].plot(); plt.xlabel('Time'); plt.ylabel('Infected'); plt.title('SI Curve')
    plt.savefig(out_png, dpi=300); plt.close()
    print("[analyze] SI summary + plot saved")


def plot_SIR(history, out_png='results/figures/curve_sir.png', out_csv='results/logs/summary_sir.csv'):
    df = pd.DataFrame(history, columns=['infected','recovered'])
    df.to_csv(out_csv, index=False)
    plt.figure(); df.plot(); plt.xlabel('Time'); plt.title('SIR Curve'); plt.savefig(out_png, dpi=300); plt.close()
    print("[analyze] SIR summary + plot saved")


if __name__ == '__main__':
    key = 'history_SIR' if config['recovery_prob']>0 else 'history_SI'
    path = f'results/logs/{key}.pkl'
    try:
        history = load_pickle(path)
    except FileNotFoundError:
        history = load_pickle('results/logs/history_SI.pkl')
        print(f"[analyze] fallback to SI history")
    if isinstance(history[0], tuple): plot_SIR(history)
    else: plot_SI(history)