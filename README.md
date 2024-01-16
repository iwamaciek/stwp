# STWP - Short Term Weather Prediction


This repository is specifically designed for the research component of the engineering thesis titled "A Machine Learning System for Short-Term Weather Prediction." It encompasses not only the implementation of baseline models and the main architecture but also includes modules for data preprocessing, training pipelines, hyperparameter optimization, result presentation, an API facilitating communication with a [mobile application](https://github.com/JaJasiok/meteo-mind/) for the best model, and scripts for downloading data from the [ERA5](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels) dataset.

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
pre-commit install
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
