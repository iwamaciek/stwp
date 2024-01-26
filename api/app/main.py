import shutil
import json
import os
import zipfile
from fastapi import FastAPI, Query
from fastapi.responses import FileResponse
from pprint import pprint
import math
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from datetime import datetime, timedelta
import sys
sys.path.append("..")
from model.utils.get_data import DataImporter
from model.config import config as cfg
from model.trainer import Trainer
import cartopy.io.shapereader as shpreader
import matplotlib.path as mpath
import matplotlib.patches as patches


# Function to create new predictions
def get_current_data():
    global json_data
    current_date = datetime.now() - timedelta(days=7) #timedelta because new data is unavailable - we use the data from the week before

    # This section downloads the data
    previous_day = current_date - timedelta(days=1)

    if previous_day.year != current_date.year:
        dataImporter.query_dict["year"] = [f"{previous_day.year}", f"{current_date.year}"]
    else:
        dataImporter.query_dict["year"] = [f"{current_date.year}"]

    if previous_day.month != current_date.month:
        dataImporter.query_dict["month"] = [f"{previous_day.month}", f"{current_date.month}"]
    else:
        dataImporter.query_dict["month"] = [f"{current_date.month}"]

    if previous_day.day > current_date.day:
        dataImporter.query_dict["day"] = [f"0{current_date.day}", f"{previous_day.day}"]
    else:
        if previous_day.day < 10:
            previous_day = f"0{previous_day.day}"
        else:
            previous_day = f"{previous_day.day}"
        if current_date.day < 10:
            current_day = f"0{current_date.day}"
        else:
            current_day = f"{current_date.day}"
        dataImporter.query_dict["day"] = [f"{previous_day}", f"{current_day}"]

    most_recent_hour = (current_date.hour // 6) * 6
    dataImporter.query_dict["hour"] = ["00:00", "06:00", "12:00", "18:00"]

    dataImporter.download_data()

    # This section creates the prediction and saves it into a JSON file
    trainer = Trainer(architecture="trans", hidden_dim=32)
    trainer.load_model("../model/data/gnn_fh5.pt")

    if most_recent_hour == 0:
        json_data = trainer.predict_to_json(which_sequence=0)
    elif most_recent_hour == 6:
        json_data = trainer.predict_to_json(which_sequence=1)
    elif most_recent_hour == 12:
        json_data = trainer.predict_to_json(which_sequence=2)
    elif most_recent_hour == 18:
        json_data = trainer.predict_to_json(which_sequence=3)
    else:
        raise ValueError(f"Hour is incompatible: {most_recent_hour}")

    current_date = datetime(year=current_date.year, month=current_date.month, day=current_date.day, hour=most_recent_hour)

    return current_date, json_data


# Initialize FastAPI app
app = FastAPI()

# Load JSON data from file
# with open("./data/data3.json", "r") as file:
#     json_data = json.load(file)
json_data = None

# Create new data
dataImporter = DataImporter()
cfg.DATA_PATH = "../model/data/data.grib"
cfg.TRAIN_RATIO = 0
cfg.BATCH_SIZE = 1
cfg.FH = 5
cfg.INPUT_SIZE = 5
previous_data_gather, json_data = get_current_data()

# Get the minimum and maximum latitude and longitude from the data
lat_min = float(min(json_data.keys()))
lat_max = float(max(json_data.keys()))
lng_min = float(min(json_data[str(lat_min)].keys()))
lng_max = float(max(json_data[str(lat_min)].keys()))

# Set the coordinate accuracy
coord_acc = 0.25

# Function to calculate the fractions used for interpolation
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


# Function to interpolate the value for a given latitude and longitude
def interpolate_value(array, lat, lng):
    lat_index = int(np.floor((lat_max - lat) / coord_acc))
    lng_index = int(np.floor((lng - lng_min) / coord_acc))

    # Calculate the fractions used for interpolation
    lat_frac, lng_frac = get_fractions(lat, lng)

    # Handle edge cases where lat or lng are at their maximum values
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
        # Interpolate the value
        return (
            (1 - lat_frac) * (1 - lng_frac) * array[lat_index, lng_index]
            + lat_frac * (1 - lng_frac) * array[lat_index, lng_index + 1]
            + (1 - lat_frac) * lng_frac * array[lat_index + 1, lng_index]
            + lat_frac * lng_frac * array[lat_index + 1, lng_index + 1]
        )


# Function to get the values for a given latitude and longitude
def get_values_by_lat_lng(lat, lng):
    global json_data

    # Get all timestamps from the json data
    timestamps = json_data["55.0"]["14.0"]["t2m"].keys()

    # Create a dictionary of features with their respective values
    # Each feature is a 3D numpy array with dimensions corresponding to latitude, longitude, and time
    features = {
        "sp": np.array([[[json_data[lat][lng]["sp"][timestamp] for timestamp in json_data[lat][lng]["sp"]] for lng in json_data[lat]] for lat in json_data]),
        "tcc": np.array([[[json_data[lat][lng]["tcc"][timestamp] if 0 <= json_data[lat][lng]["tcc"][timestamp] <= 1 else 0 if json_data[lat][lng]["tcc"][timestamp] < 0 else 1 for timestamp in json_data[lat][lng]["tcc"]] for lng in json_data[lat]] for lat in json_data]),
        "tp": np.array([[[json_data[lat][lng]["tp"][timestamp] for timestamp in json_data[lat][lng]["tp"]] for lng in json_data[lat]] for lat in json_data]),
        "u10": np.array([[[json_data[lat][lng]["u10"][timestamp] * 3.6 for timestamp in json_data[lat][lng]["u10"]] for lng in json_data[lat]] for lat in json_data]),
        "v10": np.array([[[json_data[lat][lng]["v10"][timestamp] * 3.6 for timestamp in json_data[lat][lng]["v10"]] for lng in json_data[lat]] for lat in json_data]),
        "t2m": np.array([[[json_data[lat][lng]["t2m"][timestamp] for timestamp in json_data[lat][lng]["t2m"]] for lng in json_data[lat]] for lat in json_data]),
    }

    # Initialize the response dictionary with latitude, longitude, and an empty list of timestamps
    response = {"lat": lat, "lng": lng, "timestamps": []}

    # Loop through each timestamp
    for i, timestamp in enumerate(timestamps):
        data = {"timestamp": timestamp, "values": {}}
        for feature in features:
            value = float(interpolate_value(
                features[feature][:, :, i], lat, lng))
            data["values"][feature] = value
        response["timestamps"].append(data)

    return response


# Function to create maps for different weather features
def create_maps():
    global json_data

    # Define the latitude and longitude ranges
    lats = np.arange(lat_max, lat_min - coord_acc, -coord_acc)
    lons = np.arange(lng_min, lng_max + coord_acc, coord_acc)

    # Define the latitude and longitude ranges
    features = {
        "tcc": np.array([[[json_data[lat][lng]["tcc"][timestamp] if 0 <= json_data[lat][lng]["tcc"][timestamp] <= 1 else 0 if json_data[lat][lng]["tcc"][timestamp] < 0 else 1 for timestamp in json_data[lat][lng]["tcc"]] for lng in json_data[lat]] for lat in json_data]),
        "tp": np.array([[[json_data[lat][lng]["tp"][timestamp] if 0.05 <= json_data[lat][lng]["tp"][timestamp] <= 100 else 0 if json_data[lat][lng]["tp"][timestamp] < 0.05 else 100 for timestamp in json_data[lat][lng]["tp"]] for lng in json_data[lat]] for lat in json_data]),
        "t2m": np.array([[[json_data[lat][lng]["t2m"][timestamp] for timestamp in json_data[lat][lng]["t2m"]] for lng in json_data[lat]] for lat in json_data]),
    }

    # Define the color maps for the different features
    colors = {"tcc": "Greens", "t2m": "coolwarm"}

    t2m_min = math.ceil(np.min(features["t2m"]))
    t2m_max = math.floor(np.max(features["t2m"]))

    # Define the ranges for the different features
    ranges = {
        "tp": [0.1, 0.25, 0.5, 1, 2.5, 5, 7.5, 10, 15, 20, 30, 40,
               50, 100],
        "tcc": np.arange(0.01, 1.01, 0.01),
        "t2m": np.arange(t2m_min - 1, t2m_max + 2, 1),
    }

    custom_colors = [(0.8431372549, 0.91764705882, 0.97647058823),
                     (0.3137255012989044, 0.8156862854957581,
                     0.8156862854957581),
                     (0.0, 1.0, 1.0),
                     (0.0, 0.8784313797950745, 0.501960813999176),
                     (0.0, 0.7529411911964417, 0.0),
                     (0.501960813999176, 0.8784313797950745, 0.0),
                     (1.0, 1.0, 0.0),
                     (1.0, 0.6274510025978088, 0.0),
                     (1.0, 0.0, 0.0),
                     (1.0, 0.125490203499794, 0.501960813999176),
                     (0.9411764740943909, 0.250980406999588, 1.0),
                     (0.501960813999176, 0.125490203499794, 1.0),
                     (0.250980406999588, 0.250980406999588, 1.0)]

    labels = {
        "tp" : "[mm]",
        "tcc" : "[%]",
        "t2m" : "[°C]"
    }

    # Get the shape of Poland
    for country in shpreader.Reader(shpreader.natural_earth(resolution='10m', category='cultural', name='admin_0_countries')).records():
        if country.attributes['NAME_LONG'] == 'Poland':
            poland_geom = country.geometry
            break

    # Convert the polygon to a path
    poland_vertices = list(poland_geom.exterior.coords)

    poland_path = mpath.Path(poland_vertices)

    # Loop through each feature
    for feature_name, feature in features.items():
        for i in range(feature.shape[2]):
            data = feature[:, :, i]
            map_crs = ccrs.Mercator(central_longitude=40)
            data_crs = ccrs.PlateCarree()

            # Create a new figure for the map
            fig = plt.figure(1, figsize=(14, 12))

            # Add a subplot with the map coordinate reference system
            ax = plt.subplot(1, 1, 1, projection=map_crs)
            ax.set_extent([14, 25, 49, 55])
            # ax.add_feature(cfeature.COASTLINE.with_scale("50m"))
            # ax.add_feature(cfeature.BORDERS)

            levels = ranges.get(feature_name)

            # Define the color map for the current feature and add a filled contour plot of the data
            if feature_name == "tp":
                cmap = mcolors.ListedColormap(custom_colors, "custom_cmap")
                norm = mcolors.BoundaryNorm(ranges["tp"], cmap.N)

                cf = ax.contourf(
                    lons, lats, data, levels=levels, cmap=cmap, transform=data_crs, norm=norm
                )
            else:
                cmap = plt.colormaps[colors.get(feature_name)]
                cf = ax.contourf(
                    lons, lats, data, levels=levels, cmap=cmap, transform=data_crs
                )

            # Create a mask for the area outside Poland
            patch = patches.PathPatch(poland_path, transform=ccrs.PlateCarree(), facecolor='none')
            ax.add_patch(patch)
            for collection in cf.collections:
                collection.set_clip_path(patch)

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
        if feature_name == "t2m":
            ticks = np.linspace(min_val, max_val, num=(
                len(ranges[feature_name])))
        elif feature_name == "tcc":
            ticks = np.linspace(min_val, max_val, num=11)
        else:
            ticks = ranges["tp"]

        # Set the ticks
        cbar.set_ticks(ticks)

        if feature_name == "tcc":
            ticklabels = [f"{int(tick * 100)}" for tick in ticks]
        elif feature_name == "t2m":
            ticklabels = [f"{int(tick)}" for tick in ticks]
        else:
            ticklabels = [f"{tick}" for tick in ticks]
            ticklabels[-1] = "≥" + str(int(ticks[-1]))

        # Set the label
        cbar.set_label(labels[feature_name], fontsize=32, color='white')

        # Set the tick labels
        cbar.set_ticklabels(ticklabels, fontsize=32, color='white')

        # Move the plot around
        plt.subplots_adjust(left=0.03, right=0.97, top=0.9, bottom=0.4)

        plt.savefig(f"./maps/{feature_name}_legend.png", transparent=True)

        plt.clf()


# Define endpoint for weather data
@app.get("/weather")
async def get_weather(
    latitude: float = Query(..., description="Latitude of the location"),
    longitude: float = Query(..., description="Longitude of the location"),
):
    global json_data
    global previous_data_gather
    current_date = datetime.now() - timedelta(days=7) #timedelta because new data is unavailable - we use the data from the week before
    if((current_date - previous_data_gather).total_seconds() > 21600): # 6 hours
        # Get new data - not implemented yet
        previous_data_gather, json_data = get_current_data()
    return get_values_by_lat_lng(latitude, longitude)


# Define endpoint for maps
@app.get("/maps")
async def get_maps():
    global json_data
    global previous_data_gather
    current_date = datetime.now() - timedelta(days=7) #timedelta because new data is unavailable - we use the data from the week before
    if((current_date - previous_data_gather).total_seconds() > 21600): # 6 hours
        # Get new data - not implemented yet
        previous_data_gather, json_data = get_current_data()
        create_maps()
    # Get a list of all files in the "maps" folder
    images = os.listdir("./maps")

    # Create a Zip file
    with open("./maps.zip", "wb") as f_out:
        with zipfile.ZipFile(f_out, mode="w") as archive:
            for image in images:
                full_path = os.path.join("./maps", image)
                archive.write(full_path, arcname=os.path.join("maps", image))

    return FileResponse("./maps.zip", media_type="application/zip")

@app.get("/info")
async def get_info():
    info_dict = {
        "current_data_from": previous_data_gather,
        "status": "operational",
    }
    return info_dict


# Main function
if __name__ == "__main__":
    import uvicorn

    create_maps()

    uvicorn.run(app, host="0.0.0.0", port=8888)
