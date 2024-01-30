#!/bin/bash
apt update -y
apt install g++ libeccodes-dev -y
pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cpu
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-2.1.2-cpu.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-2.1.2-cpu.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-2.1.2-cpu.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-2.1.2-cpu.html
pip install torch-geometric
pip install torch-geometric-temporal