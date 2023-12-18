import shutil
import os
import zipfile
from fastapi import FastAPI, Query
from fastapi.responses import FileResponse
import cfgrib
from pprint import pprint
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt

app = FastAPI()

grib_data = cfgrib.open_datasets("2022_10_18.grib")

lat_min = min(grib_data[0].coords["latitude"].values)
lat_max = max(grib_data[0].coords["latitude"].values)
lng_min = min(grib_data[0].coords["longitude"].values)
lng_max = max(grib_data[0].coords["longitude"].values)

coord_acc = 0.25


def get_fractions(lat, lng):

    latitudes = np.arange(lat_max, lat_min - coord_acc, -coord_acc)
    longitudes = np.arange(lng_min, lng_max + coord_acc, coord_acc)

    lat_index = int(np.floor((lat_max - lat) / coord_acc))
    lng_index = int(np.floor((lng - lng_min) / coord_acc))

    lat_center = latitudes[lat_index]
    long_center = longitudes[lng_index]

    lat_distance = (lat_center - lat) / coord_acc
    long_distance = (lng - long_center) / coord_acc

    return lat_distance, long_distance


def interpolate_value(array, lat, lng):
    lat_index = int(np.floor((lat_max - lat) / coord_acc))
    lng_index = int(np.floor((lng - lng_min) / coord_acc))

    lat_frac, lng_frac = get_fractions(lat, lng)

    # lat_frac = (lat - lat_min) * 4 - lat_index
    # lng_frac = (lng - lng_min) * 4 - lng_index

    if lat == lat_max and lng == lng_max:
        return array[lat_index, lng_index]
    elif lat == lat_max:
        return (1 - lng_frac) * array[lat_index, lng_index] + lng_frac * array[
            lat_index, lng_index + 1
        ]
    elif lng == lng_max:
        return (1 - lat_frac) * array[lat_index, lng_index] + lat_frac * array[
            lat_index + 1, lng_index
        ]
    else:
        return (
            (1 - lat_frac) * (1 - lng_frac) * array[lat_index, lng_index]
            + lat_frac * (1 - lng_frac) * array[lat_index, lng_index + 1]
            + (1 - lat_frac) * lng_frac * array[lat_index + 1, lng_index]
            + lat_frac * lng_frac * array[lat_index + 1, lng_index + 1]
        )


def get_values_by_lat_lng(lat, lng):

    surface = grib_data[0]
    hybrid = grib_data[1]

    timestamps = surface.time.values.astype(str)

    tp = hybrid.tp.to_numpy() * 1000
    if tp.ndim >= 4:
        tp = tp.reshape((-1,) + hybrid.tp.shape[2:])

    features = {
        "sp": surface.sp.to_numpy() / 100,
        "tcc": surface.tcc.to_numpy(),
        "tp": tp,
        "u10": surface.u10.to_numpy() * 3.6,
        "v10": surface.v10.to_numpy() * 3.6,
        "t2m": surface.t2m.to_numpy() - 273.15,
    }

    response = {"lat": lat, "lng": lng, "timestamps": []}

    for i, timestamp in enumerate(timestamps):
        data = {"timestamp": timestamp, "values": {}}
        for feature in features:
            value = round(float(interpolate_value(features[feature][i], lat, lng)), 2)
            data["values"][feature] = value
        response["timestamps"].append(data)

    return response


def create_maps():
    surface = grib_data[0]
    hybrid = grib_data[1]

    tp = hybrid.tp.to_numpy() * 1000
    if tp.ndim >= 4:
        tp = tp.reshape((-1,) + hybrid.tp.shape[2:])

    # pprint(tp)

    lats = surface.coords["latitude"].values
    lons = surface.coords["longitude"].values

    features = {
        "tp": tp,
        "tcc": surface.tcc.to_numpy(),
        "t2m": surface.t2m.to_numpy() - 273.15,
    }

    colors = {"tp": "Blues", "tcc": "Greens", "t2m": "coolwarm"}

    ranges = {
        "tp": np.arange(0, 20.1, 0.1),
        "tcc": np.arange(0, 1.01, 0.01),
        "t2m": np.arange(-20, 41, 1),
    }

    for feature_name, feature in features.items():
        for i, data in enumerate(feature):
            map_crs = ccrs.Mercator(central_longitude=40)
            data_crs = ccrs.PlateCarree()

            fig = plt.figure(1, figsize=(14, 12))

            ax = plt.subplot(1, 1, 1, projection=map_crs)
            ax.set_extent([14, 25, 49, 55])
            ax.add_feature(cfeature.COASTLINE.with_scale("50m"))
            ax.add_feature(cfeature.BORDERS)

            gl = ax.gridlines(draw_labels=False, linewidth=0, linestyle="--")
            levels = ranges.get(feature_name)
            cmap = plt.colormaps[colors.get(feature_name)]
            cmap.set_bad(alpha=0)  # Set white/zero values as transparent

            cf = ax.contourf(
                lons, lats, data, levels=levels, cmap=cmap, transform=data_crs
            )

            ax.add_feature(cfeature.BORDERS)

            # Remove axis and labels
            ax.set_axis_off()

            # Save the map without axis and labels
            if not os.path.exists("./maps"):
                os.mkdir("maps")
            plt.savefig(
                f"./maps/{feature_name}" + str(i + 1) + ".png",
                bbox_inches="tight",
                pad_inches=0,
            )

            plt.clf()


@app.get("/weather")
async def get_weather(
    latitude: float = Query(..., description="Latitude of the location"),
    longitude: float = Query(..., description="Longitude of the location"),
):
    result = get_values_by_lat_lng(latitude, longitude)
    return result


@app.get("/maps")
async def get_maps():
    # create_maps()
    images = os.listdir("maps")  # Get a list of all files in the "maps" folder

    # Create a Zip file
    with open("maps.zip", "wb") as f_out:
        with zipfile.ZipFile(f_out, mode="w") as archive:
            for image in images:
                archive.write(os.path.join("maps", image))

    return FileResponse("maps.zip", media_type="application/zip")


# http://127.0.0.1:8000/weather?latitude=49.7128&longitude=21.006

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)

    # pprint(grib_data[0].u10)
    # pprint(grib_data[0].v10)

    # plt.imshow(grib_data[0].tcc[2])

    # plt.show()
    # create_maps()

    # pprint(grib_data)

    # val = get_values_by_lat_lng(grib_data, 55.75, 14.25)
    # paint_map(grib_data)

    # pprint(val)
