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

### Feature Examination
The examined features include key meteorological parameters, each contributing to a comprehensive understanding of atmospheric conditions. The table below outlines these features along with their respective symbols, quantities, and units:

| Symbol | Quantity                             | Unit     |
| ------ | ------------------------------------ | -------- |
| t2m    | Temperature at 2m above ground       | °C       |
| sp     | Surface pressure                     | hPa      |
| tcc    | Total cloud cover                    | (0 - 1)  |
| u10    | 10m U wind component                 | $m/s$    |
| v10    | 10m V wind component                 | $m/s$    |
| tp     | Total precipitation                  | mm       |


### Prediction Quality Visualization

In the demonstration below, we showcase the prediction quality of the graph architecture using a randomly selected learning example. The forecasting horizon is set to 1, representing the prediction timestamp t+1, which corresponds to a projection 6 hours into the future:
![gnn_sample_pred-1](https://github.com/iwamaciek/stwp/assets/82380348/12a3a40e-4baa-4274-807d-fa742fa7d710)


