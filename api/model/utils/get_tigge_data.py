#!/usr/bin/env python
from ecmwfapi import ECMWFDataServer

server = ECMWFDataServer()

dates = "2021-01-01/to/2021-12-31"

server.retrieve(
    {
        "class": "ti",
        "dataset": "tigge",
        "date": dates,
        "expver": "prod",
        "grid": "0.25/0.25",
        "levtype": "sfc",
        "origin": "ecmf",
        "param": "134/165/166/167/228164/228228",  # sp, u10, v10, t2m, tcc, tp (jest też 228144="snowfall water eqivalent" i 228141="snow depth water eqivalent"; humidity nie ma)
        "step": "0/to/12/by/6",
        "time": "00:00:00/12:00:00",
        "area": "55/14/49/25",  # N/W/S/E
        "type": "fc",  # fc=forecast, cf=control forecast,
        "target": "../" + dates.replace("/", "-") + "_tigge.grib",
    }
)
