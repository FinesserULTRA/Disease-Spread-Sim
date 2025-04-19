import pickle
from pathlib import Path

def save_pickle(obj, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def load_pickle(path: str):
    with open(path, 'rb') as f:
        return pickle.load(f)