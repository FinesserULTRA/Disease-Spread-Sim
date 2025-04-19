import os
from src.generate import generate_network

def test_generate_creates_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    # override config for small N
    import src.config as c; c.config['N']=10
    generate_network()
    assert (tmp_path / 'data' / 'network_erdos_renyi.pkl').exists()