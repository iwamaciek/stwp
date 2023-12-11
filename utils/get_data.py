import cdsapi

c = cdsapi.Client()

dataset = "reanalysis-era5-single-levels"
path = "../data2021-2023.grib"

BIG_AREA = [55.75, 13.25, 48, 25]  # for nn
SMALL_AREA = [55, 14, 49, 25]

query_dict = {
    "product_type": "reanalysis",
    "format": "grib",
    "variable": [
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "2m_temperature",
        "geopotential",
        "land_sea_mask",
        "surface_pressure",
        "total_cloud_cover",
        "total_precipitation",
    ],
    "year": ["2020", "2021", "2022"],
    "month": [
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
        "31",
    ],
    "time": [
        "00:00",
        "06:00",
        "12:00",
        "18:00",
    ],
    "area": BIG_AREA,
}


if __name__ == "__main__":
    c.retrieve(dataset, query_dict, path)

# ~ 24min
