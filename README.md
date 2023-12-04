# meteoapp-data

### Install prerequisites:
```shell
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

##### If you want to play with torch_geometric_temporal:
```
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-{PYTORCH_VERSION}.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-{PYTORCH_VERSION}.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-{PYTORCH_VERSION}.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-{PYTORCH_VERSION}.html
pip install torch-geometric
pip install torch-geometric-temporal
```


### Config pre-commit hooks
<!-- Instruction [here](pre-commit-instruction.md). -->
```shell
pip install -r requirements.txt
pre-commits install
```
