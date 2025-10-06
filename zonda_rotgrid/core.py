import numpy as np
import xarray as xr
from pyproj import CRS, Transformer
try:
    from importlib.metadata import version as pkg_version
except ImportError:
    from importlib_metadata import version as pkg_version  # For Python <3.8

def compute_corner_offsets(step_x_deg, step_y_deg):
    half_x = step_x_deg / 2
    half_y = step_y_deg / 2
    return np.array([
        [-half_x, -half_y],  # SW
        [-half_x,  half_y],  # NW
        [ half_x,  half_y],  # NE
        [ half_x, -half_y],  # SE
    ])  # shape: (4, 2)

def create_rotated_grid(grid_spacing, center_lat, center_lon, hwidth_lat, hwidth_lon, pole_lat, pole_lon, ncells_boundary, output_path):
    # Convert grid spacing from km to degrees 
    dx_deg = grid_spacing / 111.320 
    dy_deg = grid_spacing / 110.574

    polgam = 0.0  # Default value for north pole grid longitude
    rotated_crs = CRS.from_cf({
        'grid_mapping_name': 'rotated_latitude_longitude',
        'grid_north_pole_latitude': pole_lat,
        'grid_north_pole_longitude': pole_lon,
        'north_pole_grid_longitude': polgam
    })
    geographic_crs = CRS.from_epsg(4326)
    nlat = int(round((2 * hwidth_lat) / dy_deg)) + 1 - 2 * ncells_boundary
    nlon = int(round((2 * hwidth_lon) / dx_deg)) + 1 - 2 * ncells_boundary
    rlat = np.linspace(center_lat - hwidth_lat, center_lat + hwidth_lat, nlat)
    rlon = np.linspace(center_lon - hwidth_lon, center_lon + hwidth_lon, nlon)
    rlon2d, rlat2d = np.meshgrid(rlon, rlat)
    transformer = Transformer.from_crs(rotated_crs, geographic_crs, always_xy=True)
    lon_flat, lat_flat = transformer.transform(rlon2d.flatten(), rlat2d.flatten())
    lon = lon_flat.reshape(rlon2d.shape)
    lat = lat_flat.reshape(rlon2d.shape)
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
    dummy = np.zeros_like(lat)
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
            "north_pole_grid_longitude": polgam,
        }
    )
    # Add meta information as global attributes
    try:
        version = pkg_version("zonda_rotgrid")
    except Exception:
        version = "unknown"
    ds.attrs["history"] = (
        f"Created with zonda-rotgrid v{version} on {__import__('datetime').datetime.now().isoformat()}"
    )
    ds.attrs["install_command"] = f"pip install zonda-rotgrid=={version}"
    ds.attrs["creation_command"] = (
        f"create-rotated-grid --grid_spacing {grid_spacing} --center_lat {center_lat} "
        f"--center_lon {center_lon} --hwidth_lat {hwidth_lat} --hwidth_lon {hwidth_lon} "
        f"--pole_lat {pole_lat} --pole_lon {pole_lon} --ncells_boundary {ncells_boundary} --output {output_path}"
    )
    ds.to_netcdf(output_path)
    print(f"File '{output_path}' created.")

def create_latlon_grid(grid_spacing, center_lat, center_lon, hwidth_lat, hwidth_lon, ncells_boundary, output_path):
    # Convert grid spacing from km to degrees
    dx_deg = grid_spacing / 111.320
    dy_deg = grid_spacing / 110.574
    nlat = int(round((2 * hwidth_lat) / dy_deg)) + 1 - 2 * ncells_boundary
    nlon = int(round((2 * hwidth_lon) / dx_deg)) + 1 - 2 * ncells_boundary
    lat = np.linspace(center_lat - hwidth_lat, center_lat + hwidth_lat, nlat)
    lon = np.linspace(center_lon - hwidth_lon, center_lon + hwidth_lon, nlon)
    lon2d, lat2d = np.meshgrid(lon, lat)
    corner_offsets = compute_corner_offsets(dx_deg, dy_deg)
    ny, nx = lon2d.shape
    nv = 4
    lon_vertices = np.empty((ny, nx, nv))
    lat_vertices = np.empty((ny, nx, nv))
    for i in range(nv):
        dlon = corner_offsets[i, 0]
        dlat = corner_offsets[i, 1]
        lon_vertices[:, :, i] = lon2d + dlon
        lat_vertices[:, :, i] = lat2d + dlat
    dummy = np.zeros_like(lat2d)
    ds = xr.Dataset(
        {
            "dummy": (["lat", "lon"], dummy),
            "lon_vertices": (["lat", "lon", "nv"], lon_vertices),
            "lat_vertices": (["lat", "lon", "nv"], lat_vertices),
        },
        coords={
            "lon": ("lon", lon),
            "lat": ("lat", lat),
            "nv": ("nv", np.arange(nv)),
            "lon2d": (["lat", "lon"], lon2d),
            "lat2d": (["lat", "lon"], lat2d),
        },
    )
    ds["lon"].attrs.update({
        "standard_name": "longitude",
        "long_name": "longitude",
        "units": "degrees_east"
    })
    ds["lat"].attrs.update({
        "standard_name": "latitude",
        "long_name": "latitude",
        "units": "degrees_north"
    })
    ds["lon_vertices"].attrs.update({
        "long_name": "longitude of vertices",
        "units": "degrees_east"
    })
    ds["lat_vertices"].attrs.update({
        "long_name": "latitude of vertices",
        "units": "degrees_north"
    })
    ds["dummy"].attrs.update({
        "coordinates": "lon lat",
        "grid_mapping": "latitude_longitude"
    })
    ds["latitude_longitude"] = xr.DataArray(
        0,
        attrs={
            "grid_mapping_name": "latitude_longitude"
        }
    )
    try:
        version = pkg_version("zonda_rotgrid")
    except Exception:
        version = "unknown"
    ds.attrs["history"] = (
        f"Created with zonda-rotgrid v{version} on {__import__('datetime').datetime.now().isoformat()}"
    )
    ds.attrs["install_command"] = f"pip install zonda-rotgrid=={version}"
    ds.attrs["creation_command"] = (
        f"create-latlon-grid --grid_spacing {grid_spacing} --center_lat {center_lat} "
        f"--center_lon {center_lon} --hwidth_lat {hwidth_lat} --hwidth_lon {hwidth_lon} "
        f"--ncells_boundary {ncells_boundary} --output {output_path}"
    )
    ds.to_netcdf(output_path)
    print(f"File '{output_path}' created (lat/lon grid).")