#!/usr/bin/bash
set -e

python -m src.cli generate
python -m src.cli visualize
python -m src.cli metrics
python -m src.cli simulate --model SI
python -m src.cli simulate --model SIR
python -m src.cli analyze
python -m src.cli sweep --param p_transmission --values 0.01 --values 0.03 --values 0.05 --values 0.10