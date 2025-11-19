# Install dependencies (run once in Colab)
!pip install -q pandas numpy matplotlib scipy contextily geopandas rasterio

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
import contextily as ctx
import geopandas as gpd
from matplotlib.lines import Line2D
from shapely.geometry import Point

# ---------------------------------------------------
# Load example data (replace with your file if needed)
# ---------------------------------------------------
data = pd.read_csv("timeavg_fields_grid.csv")
location_data = pd.read_csv("location_data.csv")

# Convert to GeoDataFrame in WGS84, then reproject to Web Mercator for basemap alignment
gdf = gpd.GeoDataFrame(
    data,
    geometry=[Point(lon, lat) for lon, lat in zip(data.y, data.x)],
    crs="EPSG:4326"
)
gdf_3857 = gdf.to_crs("EPSG:3857")

x, y = gdf_3857.geometry.x.values, gdf_3857.geometry.y.values
mean_vals, std_vals = gdf["mean_avg"].values, gdf["std_avg"].values

# Convert the Location data
loc_gdf = gpd.GeoDataFrame(
    location_data,
    geometry=[Point(lon, lat) for lon, lat in zip(location_data.Longitude,
                                                  location_data.Latitude)],
    crs="EPSG:4326"
).to_crs("EPSG:3857")

sds = loc_gdf[loc_gdf.Type == "SDS011"]
sta = loc_gdf[loc_gdf.Type == "STA"]


# Compute bounding box + small padding
xmin, ymin, xmax, ymax = gdf_3857.total_bounds
padx, pady = (xmax - xmin) * 0.05, (ymax - ymin) * 0.05
xmin -= padx; xmax += padx; ymin -= pady; ymax += pady

# Create regular grid
grid_res = 500
xi = np.linspace(xmin, xmax, grid_res)
yi = np.linspace(ymin, ymax, grid_res)
gx, gy = np.meshgrid(xi, yi)

def interp_and_smooth(values, sigma=3):
    grid = griddata((x, y), values, (gx, gy), method="cubic")
    if np.isnan(grid).any():
        grid = np.where(np.isnan(grid),
                        griddata((x, y), values, (gx, gy), method="nearest"),
                        grid)
    return gaussian_filter(grid, sigma=sigma)

mean_grid = interp_and_smooth(mean_vals)
std_grid = interp_and_smooth(std_vals)

# ---------------------------------------------------
# Helper: make one plot and return fig, ax
# ---------------------------------------------------
def plot_pm25_heatmap(grid, title, cmap, output_prefix):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_title(title, fontsize=14, weight="bold")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    # Add satellite basemap first
    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, crs="EPSG:3857")

    # Normalize color scale (trim outliers)
    vmin, vmax = np.nanpercentile(grid, [2, 98])
    norm = Normalize(vmin=vmin, vmax=vmax)

    # Plot heatmap FULLY opaque (alpha=0.8)
    im = ax.imshow(grid, extent=(xmin, xmax, ymin, ymax),
                   origin="lower", cmap=cmap, norm=norm, alpha=0.5)

    # --- Add station markers ---
    # SDS011 → white dots
    ax.scatter(sds.geometry.x, sds.geometry.y,
           s=40, c="white", edgecolor="black", linewidth=0.8, zorder=5)

    # STA → black crosses
    ax.scatter(sta.geometry.x, sta.geometry.y,
           s=60, c="black", marker="x", linewidth=1.2, zorder=6)


    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("PM2.5 (µg/m³)", fontsize=10)

    # Geographic (lon/lat) tick labels
    xticks = np.linspace(xmin, xmax, 5)
    yticks = np.linspace(ymin, ymax, 5)
    xticks_lon = gpd.GeoSeries([Point(x, (ymin+ymax)/2) for x in xticks], crs="EPSG:3857").to_crs("EPSG:4326").x
    yticks_lat = gpd.GeoSeries([Point((xmin+xmax)/2, y) for y in yticks], crs="EPSG:3857").to_crs("EPSG:4326").y

    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{lon:.3f}" for lon in xticks_lon])
    ax.set_xlabel("Longitude", fontsize=11)

    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{lat:.3f}" for lat in yticks_lat])
    ax.set_ylabel("Latitude", fontsize=11)

    # --- Legend for station types ---
    legend_elements = [
      Line2D([0], [0], marker='o', color='black', markerfacecolor='white',
           markersize=8, label='SDS011', linestyle=''),
      Line2D([0], [0], marker='x', color='black',
           markersize=9, label='STA', linestyle='')
    ]

    ax.legend(handles=legend_elements,
          loc='upper left',
          frameon=True,
          facecolor='white',
          framealpha=0.8,
          fontsize=10)


    plt.tight_layout()

    # Save individual figure
    png_path = f"/content/{output_prefix}.png"
    pdf_path = f"/content/{output_prefix}.pdf"
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {png_path}\n       {pdf_path}")

    return fig, ax

# ---------------------------------------------------
# Create individual plots (fully colored overlays)
# ---------------------------------------------------
fig_mean, ax_mean = plot_pm25_heatmap(mean_grid,
    title="Predicted PM₂.₅ Mean (11/11)",
    cmap="inferno",
    output_prefix="Predicted_PM25_Mean_11_11")

fig_std, ax_std = plot_pm25_heatmap(std_grid,
    title="Predicted PM₂.₅ Std (11/11)",
    cmap="inferno",
    output_prefix="Predicted_PM25_Std_11_11")

# ---------------------------------------------------
# Combined figure (side by side)
# ---------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 7))
for ax, (title, grid) in zip(
    axes,
    [("Predicted PM₂.₅ Mean (11/11)", mean_grid),
     ("Predicted PM₂.₅ Std (11/11)", std_grid)]
):
    ax.set_title(title, fontsize=14, weight="bold")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, crs="EPSG:3857")
    vmin, vmax = np.nanpercentile(grid, [2, 98])
    norm = Normalize(vmin=vmin, vmax=vmax)
    im = ax.imshow(grid, extent=(xmin, xmax, ymin, ymax),
                   origin="lower", cmap="inferno", norm=norm, alpha=0.5)

    # --- Add station markers ---
    # SDS011 → white dots
    ax.scatter(sds.geometry.x, sds.geometry.y,
           s=40, c="white", edgecolor="black", linewidth=0.8, zorder=5)

    # STA → black crosses
    ax.scatter(sta.geometry.x, sta.geometry.y,
           s=60, c="black", marker="x", linewidth=1.2, zorder=6)

    cbar = plt.colorbar(im, ax=ax, shrink=0.7)
    cbar.set_label("PM2.5 (µg/m³)", fontsize=10)
    # Convert ticks to lon/lat
    xticks = np.linspace(xmin, xmax, 5)
    yticks = np.linspace(ymin, ymax, 5)
    xticks_lon = gpd.GeoSeries([Point(x, (ymin+ymax)/2) for x in xticks], crs="EPSG:3857").to_crs("EPSG:4326").x
    yticks_lat = gpd.GeoSeries([Point((xmin+xmax)/2, y) for y in yticks], crs="EPSG:3857").to_crs("EPSG:4326").y
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{lon:.3f}" for lon in xticks_lon])
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{lat:.3f}" for lat in yticks_lat])
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    # --- Legend for station types ---
    legend_elements = [
      Line2D([0], [0], marker='o', color='black', markerfacecolor='white',
           markersize=8, label='SDS011', linestyle=''),
      Line2D([0], [0], marker='x', color='black',
           markersize=9, label='STA', linestyle='')
    ]

    ax.legend(handles=legend_elements,
          loc='upper left',
          frameon=True,
          facecolor='white',
          framealpha=0.8,
          fontsize=10)

plt.tight_layout()
combined_png = "/content/Predicted_PM25_Combined_11_11.png"
combined_pdf = "/content/Predicted_PM25_Combined_11_11.pdf"
plt.savefig(combined_png, dpi=300, bbox_inches="tight")
plt.savefig(combined_pdf, dpi=300, bbox_inches="tight")

print(f"\nCombined figure saved:\n{combined_png}\n{combined_pdf}")
