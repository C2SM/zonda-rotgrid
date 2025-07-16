#!/usr/bin/env python3
import argparse
import numpy as np
import xarray as xr
from pyproj import CRS, Transformer


def compute_corner_offsets(step_x_deg, step_y_deg):
    half_x = step_x_deg / 2
    half_y = step_y_deg / 2
    return np.array([
        [-half_x, -half_y],  # SW
        [-half_x,  half_y],  # NW
        [ half_x,  half_y],  # NE
        [ half_x, -half_y],  # SE
    ])  # shape: (4, 2)


def create_rotated_grid(dx, dy, center_lat, center_lon, hwidth_lat, hwidth_lon, pole_lat, pole_lon, ncells_boundary, output_path):
    # Convert grid spacing from km to degrees (approximate)
    degree_per_km = 1.0 / 111.2  # 1 degree ≈ 111.2 km
    dx_deg = dx * degree_per_km
    dy_deg = dy * degree_per_km

    # Define CRS
    rotated_crs = CRS.from_cf({
        'grid_mapping_name': 'rotated_latitude_longitude',
        'grid_north_pole_latitude': pole_lat,
        'grid_north_pole_longitude': pole_lon,
        'north_pole_grid_longitude': 90.0
    })
    geographic_crs = CRS.from_epsg(4326)

    # Transform center from geographic to rotated
    transformer_geo2rot = Transformer.from_crs(geographic_crs, rotated_crs, always_xy=True)
    center_rlon, center_rlat = transformer_geo2rot.transform(center_lon, center_lat)

    # Compute number of points, taking offset into account
    nlat = int(round((2 * hwidth_lat) / dy_deg)) + 1 - 2 * ncells_boundary
    nlon = int(round((2 * hwidth_lon) / dx_deg)) + 1 - 2 * ncells_boundary

    rlat = np.linspace(center_rlat - hwidth_lat, center_rlat + hwidth_lat, nlat)
    rlon = np.linspace(center_rlon - hwidth_lon, center_rlon + hwidth_lon, nlon)
    rlon2d, rlat2d = np.meshgrid(rlon, rlat)

    # Transform to geographic coordinates
    transformer = Transformer.from_crs(rotated_crs, geographic_crs, always_xy=True)
    lon_flat, lat_flat = transformer.transform(rlon2d.flatten(), rlat2d.flatten())
    lon = lon_flat.reshape(rlon2d.shape)
    lat = lat_flat.reshape(rlon2d.shape)

    # Compute corner coordinates
    corner_offsets = compute_corner_offsets(dx_deg, dy_deg)
    ny, nx = rlon2d.shape
    nv = 4
    lon_vertices = np.empty((ny, nx, nv))
    lat_vertices = np.empty((ny, nx, nv))

    for i in range(nv):
        dlon = corner_offsets[i, 0]
        dlat = corner_offsets[i, 1]
        rlon_corner = rlon2d + dlon
        rlat_corner = rlat2d + dlat
        flat_rlon = rlon_corner.flatten()
        flat_rlat = rlat_corner.flatten()
        flat_lon, flat_lat = transformer.transform(flat_rlon, flat_rlat)
        lon_vertices[:, :, i] = flat_lon.reshape((ny, nx))
        lat_vertices[:, :, i] = flat_lat.reshape((ny, nx))

    # Dummy variable
    dummy = np.zeros_like(lat)

    # Create dataset
    ds = xr.Dataset(
        {
            "lon": (["rlat", "rlon"], lon),
            "lat": (["rlat", "rlon"], lat),
            "lon_vertices": (["rlat", "rlon", "nv"], lon_vertices),
            "lat_vertices": (["rlat", "rlon", "nv"], lat_vertices),
            "dummy": (["rlat", "rlon"], dummy),
        },
        coords={
            "rlon": ("rlon", rlon),
            "rlat": ("rlat", rlat),
            "nv": ("nv", np.arange(nv)),
        },
    )

    # Add attributes
    ds["rlon"].attrs.update({
        "standard_name": "grid_longitude",
        "long_name": "rotated longitudes",
        "units": "degrees"
    })
    ds["rlat"].attrs.update({
        "standard_name": "grid_latitude",
        "long_name": "rotated latitudes",
        "units": "degrees"
    })
    ds["lon"].attrs.update({
        "standard_name": "longitude",
        "long_name": "geographical longitude",
        "units": "degrees_east",
        "bounds": "lon_vertices"
    })
    ds["lat"].attrs.update({
        "standard_name": "latitude",
        "long_name": "geographical latitude",
        "units": "degrees_north",
        "bounds": "lat_vertices"
    })
    ds["lon_vertices"].attrs.update({
        "long_name": "geographical longitude of vertices",
        "units": "degrees_east"
    })
    ds["lat_vertices"].attrs.update({
        "long_name": "geographical latitude of vertices",
        "units": "degrees_north"
    })
    ds["dummy"].attrs.update({
        "coordinates": "lon lat",
        "grid_mapping": "rotated_pole"
    })
    ds["rotated_pole"] = xr.DataArray(
        0,
        attrs={
            "long_name": "coordinates of the rotated North Pole",
            "grid_mapping_name": "rotated_latitude_longitude",
            "grid_north_pole_longitude": pole_lon,
            "grid_north_pole_latitude": pole_lat,
            "north_pole_grid_longitude": -180.0
        }
    )

    # Save to NetCDF
    ds.to_netcdf(output_path)
    print(f"✅ File '{output_path}' created.")


def main():
    parser = argparse.ArgumentParser(description="Generate a rotated coordinate grid NetCDF file for climate models.")
    parser.add_argument("--dx", type=float, required=True, help="Grid spacing in x (longitude) direction [km]")
    parser.add_argument("--dy", type=float, required=True, help="Grid spacing in y (latitude) direction [km]")
    parser.add_argument("--center_lat", type=float, required=True, help="Center latitude of the domain")
    parser.add_argument("--center_lon", type=float, required=True, help="Center longitude of the domain")
    parser.add_argument("--hwidth_lat", type=float, required=True, help="Half-width of domain in latitude [degrees]")
    parser.add_argument("--hwidth_lon", type=float, required=True, help="Half-width of domain in longitude [degrees]")
    parser.add_argument("--pole_lat", type=float, required=True, help="Rotated pole latitude")
    parser.add_argument("--pole_lon", type=float, required=True, help="Rotated pole longitude")
    parser.add_argument("--ncells_boundary", type=int, required=True, help="Lateral boundary cells to be removed")
    parser.add_argument("--output", type=str, required=True, help="Output NetCDF file path")

    args = parser.parse_args()

    create_rotated_grid(
        dx=args.dx,
        dy=args.dy,
        center_lat=args.center_lat,
        center_lon=args.center_lon,
        hwidth_lat=args.hwidth_lat,
        hwidth_lon=args.hwidth_lon,
        pole_lat=args.pole_lat,
        pole_lon=args.pole_lon,
        ncells_boundary=args.ncells_boundary,
        output_path=args.output
    )


if __name__ == "__main__":
    main()
