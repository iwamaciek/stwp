import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature


def draw_poland(ax, X, title, cmap, **spatial):
    lat_span = spatial["lat_span"]
    lon_span = spatial["lon_span"]
    spatial_limits = spatial["spatial_limits"]
    ax.set_extent(spatial_limits)
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"))
    ax.add_feature(cfeature.BORDERS)
    contour_plot = ax.contourf(
        lon_span, lat_span, X, transform=ccrs.PlateCarree(), cmap=cmap
    )
    ax.set_title(title)
    plt.colorbar(contour_plot)
