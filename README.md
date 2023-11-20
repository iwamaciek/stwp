# meteoapp-data

### Install prerequisites:
```shell
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
### Config pre-commit hooks
<!-- Instruction [here](pre-commit-instruction.md). -->
```shell
pip install -r requirements.txt
pre-commits install
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