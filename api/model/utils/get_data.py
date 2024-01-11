import cdsapi
BIG_AREA = [55.75, 13.25, 48, 25]  # for nn
SMALL_AREA = [55, 14, 49, 25]

class DataImporter():
    def __init__(self):
        self.c = cdsapi.Client()


        self.dataset = "reanalysis-era5-single-levels"
        self.path = "../model/data/data.grib"

        self.query_dict = {
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
            "year": ["2023"],
            "month": [
                "12",
            ],
            "day": [
                "01",
                "02",
            ],
            "time": [
                "00:00",
                "06:00",
                "12:00",
                "18:00",
            ],
            "area": BIG_AREA,
        }

    def download_data(self):
        self.c.retrieve(self.dataset, self.query_dict, self.path)


# if __name__ == "__main__":
#     download_data()
    

# ~ 24min
