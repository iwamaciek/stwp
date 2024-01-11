# stwp - short term weather prediction

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
pre-commit install
```

### Hyperparameter optimization
For linear regression:
```shell
python hpo_linear_regression.py <baseline_type: linear or simple-linear> <number of trials> <use neighbours: True or False>
```
Sample:
```shell
python hpo_linear_regression.py linear 5 False
```

### API dockerization
Create image:
```shell
docker build -t meteo-api ./api
```
Run the container
```shell
docker run -it --rm -p 8080:8888 --name api meteo-api
```
