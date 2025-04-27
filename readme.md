# Project: Simulating Infectious Disease Spread in a Social Network

A turnkey pipeline for generating synthetic or ingesting real-world social networks, visualizing them, simulating SI/SIR disease spread, analyzing results, and producing deliverables.

---

## 📂 Directory Structure
```
project_root/
├─ README.md                  # Project overview and quickstart
├─ requirements.txt           # Python dependencies
├─ config.json                # User-configurable parameters
├─ Dockerfile                 # Container definition
├─ data/                      # Saved NetworkX graphs
│  ├─ network_erdos_renyi.pkl
│  ├─ network_watts_strogatz.pkl
│  ├─ network_barabasi_albert.pkl
│  └─ network_stochastic_block.pkl
├─ src/                       # Source modules
│  ├─ config.py               # Load defaults & user overrides
│  ├─ utils.py                # I/O helpers
│  ├─ generate.py             # Synthetic generation & ingestion
│  ├─ visualize.py            # Force-directed plotting
│  ├─ metrics.py              # Degree & centrality analysis
│  ├─ simulate.py             # SI & SIR simulation
│  ├─ analyze.py              # Infection curve plotting & CSV summary
│  ├─ sweep.py                # Parameter sweep experiments
│  └─ cli.py                  # Click-based command-line interface
├─ results/                   # Outputs
│  ├─ figures/                # PNG plots
│  └─ logs/                   # Pickled histories, CSV summaries
├─ tests/                     # PyTest tests
│  ├─ test_generate.py
│  └─ test_simulate.py
├─ notebooks/                 # Jupyter exploration notebooks
│  └─ exploration.ipynb
└─ report/                    # LaTeX deliverables
   ├─ report.tex              # Detailed write-up
   └─ slides.tex              # Beamer presentation
```

---

## README.md

# Simulating Infectious Disease Spread in a Social Network

Features:
- Synthetic network models: ER, WS, BA, SBM
- Real-world network ingestion (edge lists or CSV)
- CLI & optional Docker container
- Visualization: force-directed layout, degree histograms, centrality heatmaps
- Epidemic models: SI and SIR, with parameter sweeping
- Metrics: infection curves, peak stats, network metrics (degree, centrality)
- Unit tests and Jupyter notebooks for exploratory analysis
- LaTeX report and Beamer slides

## Quickstart

```bash
# Clone and install dependencies
git clone <repo_url> && cd <repo_dir>
python3 -m venv env; source env/bin/activate
pip install -r requirements.txt

# (Optional) Docker build
docker build -t disease-sim .
```

### Examples
```bash
# Ingest real-world network
python -m src.cli ingest --input path/to/edgelist.csv
# Generate synthetic network
python -m src.cli generate --model erdos_renyi
# Visualize and compute metrics
python -m src.cli visualize --input data/network_real.pkl
python -m src.cli metrics --input data/network_real.pkl
# Simulate SI or SIR
python -m src.cli simulate --model SI
python -m src.cli simulate --model SIR
# Analyze results
python -m src.cli analyze
# Parameter sweep
python -m src.cli sweep --param p_transmission --values 0.01 --values 0.03 --values 0.05
# Run tests
pytest
# Compile reports
pdflatex report/report.tex
pdflatex report/slides.tex
```

---
