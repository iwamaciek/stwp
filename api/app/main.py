import shutil
import json
import os
import zipfile
from fastapi import FastAPI, Query
from fastapi.responses import FileResponse
from pprint import pprint
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import sys
sys.path.append("..")
from model.utils.get_data import DataImporter
from model.config import config as cfg

app = FastAPI()

current_dir = os.getcwd()
print(current_dir)

with open("./data/data.json","r") as file:
    json_data = json.load(file)

lat_min = float(min(json_data.keys()))
lat_max = float(max(json_data.keys()))
lng_min = float(min(json_data[str(lat_min)].keys()))
lng_max = float(max(json_data[str(lat_min)].keys()))

coord_acc = 0.25

previous_data_gather = datetime.now()
dataImporter = DataImporter()
dataImporter.download_data()
cfg.DATA_PATH = "../model/data/data-example.grib"
cfg.TRAIN_RATIO = 0

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

def custom_formatter(x):
    if x == int(x) and x != 1:
        return f"{int(x)}"
    elif 0 <= x <= 1:
        return f"{int(x * 100)}%"
    else:
        return f"{x}"


def interpolate_value(array, lat, lng):
    lat_index = int(np.floor((lat_max - lat) / coord_acc))
    lng_index = int(np.floor((lng - lng_min) / coord_acc))

    lat_frac, lng_frac = get_fractions(lat, lng)

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

    # lat, lng = str(lat), str(lng)

    timestamps = json_data["55.0"]["14.0"]["t2m"].keys()

    features = {
        "sp": np.array([[[json_data[lat][lng]["sp"][timestamp] for timestamp in json_data[lat][lng]["sp"]] for lng in json_data[lat]] for lat in json_data]),
        "tcc": np.array([[[json_data[lat][lng]["tcc"][timestamp] for timestamp in json_data[lat][lng]["tcc"]] for lng in json_data[lat]] for lat in json_data]),
        "tp": np.array([[[json_data[lat][lng]["tp"][timestamp] * 1000 for timestamp in json_data[lat][lng]["tp"]] for lng in json_data[lat]] for lat in json_data]),
        "u10": np.array([[[json_data[lat][lng]["u10"][timestamp] * 3.6 for timestamp in json_data[lat][lng]["u10"]] for lng in json_data[lat]] for lat in json_data]),
        "v10": np.array([[[json_data[lat][lng]["v10"][timestamp] * 3.6 for timestamp in json_data[lat][lng]["v10"]] for lng in json_data[lat]] for lat in json_data]),
        "t2m": np.array([[[json_data[lat][lng]["t2m"][timestamp] for timestamp in json_data[lat][lng]["t2m"]] for lng in json_data[lat]] for lat in json_data]),
    }

    # lat, lng = float(lat), float(lng)

    response = {"lat": lat, "lng": lng, "timestamps": []}

    for i, timestamp in enumerate(timestamps):
        data = {"timestamp": timestamp, "values": {}}
        for feature in features:
            value = float(interpolate_value(features[feature][:, :, i], lat, lng))
            data["values"][feature] = value
        response["timestamps"].append(data)

    return response


def create_maps():
    lats = np.arange(lat_max, lat_min - coord_acc, -coord_acc)
    lons = np.arange(lng_min, lng_max + coord_acc, coord_acc)

    features = {
        "tcc": np.array([[[json_data[lat][lng]["tcc"][timestamp] for timestamp in json_data[lat][lng]["tcc"]] for lng in json_data[lat]] for lat in json_data]),
        "tp": np.array([[[json_data[lat][lng]["tp"][timestamp] * 1000 for timestamp in json_data[lat][lng]["tp"]] for lng in json_data[lat]] for lat in json_data]),
        "t2m": np.array([[[json_data[lat][lng]["t2m"][timestamp] for timestamp in json_data[lat][lng]["t2m"]] for lng in json_data[lat]] for lat in json_data]),
    }
    colors = {"tp": "Blues", "tcc": "Greens", "t2m": "coolwarm"}

    ranges = {
        "tp": np.arange(0, 20.1, 0.1),
        "tcc": np.arange(0, 1.01, 0.01),
        "t2m": np.arange(-20, 41, 1),
    }

    for feature_name, feature in features.items():
        for i in range(feature.shape[2]):
            data = feature[:, :, i]
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
                f"./maps/{feature_name}" + str(i) + ".png",
                bbox_inches="tight",
                pad_inches=0,
                transparent=True,
            )

            plt.clf()

        # Create a new figure for the colorbar
        fig, ax = plt.subplots(figsize=(24, 3))
        cbar = plt.colorbar(cf, cax=ax, orientation='horizontal')

        # Calculate min and max
        min_val = np.min(ranges[feature_name])
        max_val = np.max(ranges[feature_name])

        # Generate a sequence of numbers from min to max
        ticks = np.linspace(min_val, max_val, num=11)

        # Set the ticks
        cbar.set_ticks(ticks)

        # Set the tick labels and increase font size
        cbar.set_ticklabels([custom_formatter(tick) for tick in ticks], fontsize=32, color='white')

        plt.subplots_adjust(left=0.03, right=0.97, top=0.8, bottom=0.2)

        plt.savefig(f"./maps/{feature_name}_legend.png", transparent=True)

        plt.clf()


@app.get("/weather")
async def get_weather(
    latitude: float = Query(..., description="Latitude of the location"),
    longitude: float = Query(..., description="Longitude of the location"),
):
    if((datetime.now() - previous_data_gather).seconds >= 21600): # 6 hours
        # Get new data - not implemented yet
        pass
    result = get_values_by_lat_lng(latitude, longitude)
    return result


@app.get("/maps")
async def get_maps():
    if((datetime.now() - previous_data_gather).seconds >= 21600): # 6 hours
        # Get new data - not implemented yet
        pass
    images = os.listdir("./maps")  # Get a list of all files in the "maps" folder

    # Create a Zip file
    with open("./maps.zip", "wb") as f_out:
        with zipfile.ZipFile(f_out, mode="w") as archive:
            for image in images:
                full_path = os.path.join("./maps", image)
                archive.write(full_path, arcname=os.path.join("maps", image))

    return FileResponse("./maps.zip", media_type="application/zip")


if __name__ == "__main__":
    import uvicorn

    create_maps()

    uvicorn.run(app, host="0.0.0.0", port=8888)
