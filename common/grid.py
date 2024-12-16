import numpy as np
import geopandas as gpd
from tqdm import tqdm
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
shapefile_path = os.path.join(current_dir, "..","layers", "municipios-regioes", "municipio.shp")

print(shapefile_path)


def load_shapefile():
    gdf = gpd.read_file(shapefile_path)
    gdf = transform_crs(gdf, 4674, 3857)
    return gdf


def transform_crs(gdf, crs_origin, crs_dest):
    gdf.set_crs(epsg=crs_origin, inplace=True)
    return gdf.to_crs(crs_dest)


def select_municipalities(municipalities):
    gdf = load_shapefile()
    selected_gdf = gdf[gdf["NM_MUN"].isin(municipalities)]
    return selected_gdf


def create_grid(grid_size, municipalities=None):
    gdf = load_shapefile()
    gdf = select_municipalities(municipalities) if municipalities else gdf

    xmin, ymin, xmax, ymax = gdf.total_bounds
    xmin, ymin, xmax, ymax = (
        xmin - grid_size,
        ymin - grid_size,
        xmax + grid_size,
        ymax + grid_size,
    )
    rows = np.arange(xmin, xmax, grid_size)
    cols = np.arange(ymin, ymax, grid_size)

    grid = []
    cell_id = 0
    for yi, y in enumerate(tqdm(rows[:-1])):
        for xi, x in enumerate(cols[:-1]):
            polygon = Polygon(
                [
                    (y, x),
                    (y + grid_size, x),
                    (y + grid_size, x + grid_size),
                    (y, x + grid_size),
                ]
            )
            grid.append(
                {"cell": cell_id, "geometry": polygon, "xcell": xi, "ycell": yi}
            )
            cell_id += 1

    grid = gpd.GeoDataFrame(grid, crs=gdf.crs)

    return grid


def create_matrix(data, grid):
    matrix = np.zeros((grid["ycell"].max() + 1, grid["xcell"].max() + 1))
    gdf = gpd.sjoin(data, grid)
    gdf = gdf.groupby(["cell", "xcell", "ycell"]).size()

    gdf = gdf.reset_index()
    gdf.columns = ["cell", "xcell", "ycell", "count"]
    for _, row in gdf.iterrows():
        matrix[row["ycell"]][row["xcell"]] = row["count"]

    return matrix


def matrix_to_grid(matrix, grid):
    aux = grid.copy()
    aux["count"] = 0
    aux["count"] = (
        matrix.flatten()
    )  # Assign the flattened matrix values to the "count" column
    aux["count"] = aux["count"] / aux["count"].sum()
    return aux


def points_in_municipalities(municipalities, data):
    points = gpd.GeoDataFrame(
        data,
        geometry=gpd.points_from_xy(data["numero_longitude"], data["numero_latitude"]),
    )
    points = transform_crs(points, 4326, 3857)

    grid = select_municipalities(municipalities)

    return gpd.sjoin(points, grid, predicate="within")


def plot_grid(grid, points=None, municipalities=None, **kwargs):

    gdf = load_shapefile()
    gdf = select_municipalities(municipalities) if municipalities else gdf

    grid = gpd.sjoin(grid, gdf)

    ax = gdf.plot(edgecolor="red", facecolor="none")

    grid.plot(ax=ax, edgecolor="black", facecolor="none", **kwargs)

    if points is not None:
        points.plot(ax=ax, color="blue")

    plt.show()


def set_points_to_grid(points, grid, column: str = "cell"):

    if column in points:
        points.drop(column, axis=1, inplace=True)

    return gpd.sjoin(points, grid, predicate="within")[column]
