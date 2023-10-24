import cdsapi

c = cdsapi.Client()

c.retrieve(
    "reanalysis-era5-single-levels",
    {
        "product_type": "reanalysis",
        "format": "grib",
        "variable": [
            "2m_temperature",
            "surface_pressure",
        ],
        "year": "2023",
        "month": [
            "09",
            "10",
        ],
        "day": [
            "01",
            "02",
            "03",
            "04",
            "05",
            "06",
            "07",
            "08",
            "09",
            "10",
            "11",
            "12",
            "13",
            "14",
            "15",
            "16",
            "17",
            "18",
            "19",
            "20",
            "21",
            "22",
            "23",
            "24",
            "25",
            "26",
            "27",
            "28",
            "29",
            "30",
        ],
        "time": [
            "00:00",
            "04:00",
            "08:00",
            "12:00",
            "16:00",
            "20:00",
        ],
        "area": [
            55,
            14,
            49,
            25,
        ],
    },
    "data.grib",
)

# ~ 30s
