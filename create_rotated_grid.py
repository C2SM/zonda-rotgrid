#!/usr/bin/env python3
import argparse
import numpy as np
import xarray as xr
from pyproj import CRS, Transformer


def compute_vertices(grid, step):
    lower = grid - step / 2
    upper = grid + step / 2
    return np.stack([
        (lower, lower),
        (lower, upper),
        (upper, upper),
        (upper, lower),
    ], axis=0).transpose((2, 3, 0))


def create_rotated_grid(dx, dy, center_lat, center_lon, hwidth_lat, hwidth_lon, pole_lat, pole_lon, output_path):
    # Compute number of grid points
    nlat = int(round((2 * hwidth_lat) / dy)) + 1
    nlon = int(round((2 * hwidth_lon) / dx)) + 1

    rlat = np.linspace(-hwidth_lat, hwidth_lat, nlat)
    rlon = np.linspace(-hwidth_lon, hwidth_lon, nlon)
    rlon2d, rlat2d = np.meshgrid(rlon, rlat)

    rotated_crs = CRS.from_cf({
        'grid_mapping_name': 'rotated_latitude_longitude',
        'grid_north_pole_latitude': pole_lat,
        'grid_north_pole_longitude': pole_lon,
        'north_pole_grid_longitude': 0.0
    })
    geographic_crs = CRS.from_epsg(4326)
    transformer = Transformer.from_crs(rotated_crs, geographic_crs, always_xy=True)

    lon_flat, lat_flat = transformer.transform(rlon2d.flatten(), rlat2d.flatten())
    lon = lon_flat.reshape(rlon2d.shape)
    lat = lat_flat.reshape(rlat2d.shape)

    nv = 4
    rlon_vertices = compute_vertices(rlon2d, dx)
    rlat_vertices = compute_vertices(rlat2d, dy)

    lon_vertices = np.empty(rlon2d.shape + (nv,))
    lat_vertices = np.empty(rlat2d.shape + (nv,))
    for i in range(nv):
        flat_lon, flat_lat = transformer.transform(
            rlon_vertices[:, :, i].flatten(),
            rlat_vertices[:, :, i].flatten()
        )
        lon_vertices[:, :, i] = flat_lon.reshape(rlon2d.shape)
        lat_vertices[:, :, i] = flat_lat.reshape(rlat2d.shape)

    dummy = np.zeros_like(lat)

    ds = xr.Dataset(
        {
            "rlon": (["rlon"], rlon),
            "rlat": (["rlat"], rlat),
            "lon": (["rlat", "rlon"], lon),
            "lat": (["rlat", "rlon"], lat),
            "lon_vertices": (["rlat", "rlon", "nv"], lon_vertices),
            "lat_vertices": (["rlat", "rlon", "nv"], lat_vertices),
            "dummy": (["rlat", "rlon"], dummy),
        },
        coords={
            "rlon": rlon,
            "rlat": rlat,
            "nv": np.arange(nv),
        },
    )

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
            "north_pole_grid_longitude": 0.0
        }
    )

    ds.to_netcdf(output_path)
    print(f"File '{output_path}' created.")


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
        output_path=args.output
    )


if __name__ == "__main__":
    main()
