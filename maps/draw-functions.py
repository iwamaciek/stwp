import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pygrib
import cartopy.crs as ccrs
import cartopy.feature as cfeature



def draw_poland(file, name, date, time):
    grbs = pygrib.open(file)
    grbs.seek(0)
    grb = grbs.select(name = name, date = date, time = time)[0]
    lats,lons = grb.latlons()

    map_crs = ccrs.Mercator(central_longitude=40)

    data_crs = ccrs.PlateCarree()

    fig = plt.figure(1,figsize=(14,12))

    ax = plt.subplot(1,1,1, projection=map_crs)
    ax.set_extent([14,25,49,55])
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'))
    ax.add_feature(cfeature.BORDERS)

    gl = ax.gridlines(
        draw_labels=True, linewidth=2, color='gray', alpha=0.5, linestyle='--'
    )
    ax.contourf(lons, lats, grb.values, transform=data_crs)