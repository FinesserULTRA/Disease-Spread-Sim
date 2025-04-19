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
        new_i, new_r = set(), set()
        for u in infected:
            if random.random() < config['recovery_prob']:
                new_r.add(u)
            else:
                for v in G.neighbors(u):
                    if v not in infected and v not in recovered and random.random() < config['p_transmission']:
                        new_i.add(v)
        infected |= new_i
        infected -= new_r
        recovered |= new_r
        history.append((len(infected), len(recovered)))
    return history, (infected, recovered)



if __name__ == '__main__':
    import os
    os.makedirs('results/logs', exist_ok=True)
    G = load_pickle(f"data/network_{config['model']}.pkl")
    if config['recovery_prob'] > 0:
        history, _ = run_SIR(G)
        save_pickle(history, f"results/logs/history_SIR.pkl")
        print("SIR simulation complete. Saved to results/logs/history_SIR.pkl")
    else:
        history, _ = run_SI(G)
        save_pickle(history, f"results/logs/history_SI.pkl")
        print("SI simulation complete. Saved to results/logs/history_SI.pkl")