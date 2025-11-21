# zonda-rotgrid

Generate regular or rotated coordinate grid NetCDF files for weather and climate models.

This package is used by [Zonda](https://zonda.ethz.ch/) for generating a lat-lon grid file
that matches the ICON grid.


## Installation

```bash
pip install zonda-rotgrid
```

## General usage

After installation, you can use the command line tool for generating rotated (`create-rotated-grid`) or regular (`create-latlon-grid`) lat/lon grids:

### Rotated grid example for EUR-12 domain (using dlon/dlat in degrees)
```
create-rotated-grid --dlon 0.11 --dlat 0.11 --center_lat -0.77 --center_lon -5.11 --hwidth_lat 22.605 --hwidth_lon 23.265 --pole_lat 39.25 --pole_lon -162.0 --ncells_boundary 0 --output EUR-12_rotated.nc
```

### Rotated grid example (using grid spacing in km)
```
create-rotated-grid --grid_spacing 12.14 --center_lat -0.77 --center_lon -5.11 --hwidth_lat 24.365 --hwidth_lon 25.025 --pole_lat 39.25 --pole_lon -162 --ncells_boundary 0 --output rotated_grid.nc
```


#### Usage
```
usage: create-rotated-grid [-h] (--grid_spacing GRID_SPACING | --dlon DLON --dlat DLAT) --center_lat CENTER_LAT --center_lon CENTER_LON --hwidth_lat HWIDTH_LAT --hwidth_lon HWIDTH_LON
                           --pole_lat POLE_LAT --pole_lon POLE_LON [--ncells_boundary NCELLS_BOUNDARY] --output OUTPUT

Generate a rotated coordinate grid NetCDF file.

optional arguments:
  -h, --help            show this help message and exit
  --grid_spacing GRID_SPACING
                        Grid spacing in horizontal direction [km]
  --dlon DLON           Grid spacing in longitude direction [degrees]
  --dlat DLAT           Grid spacing in latitude direction [degrees]
  --center_lat CENTER_LAT
                        Center latitude of the domain
  --center_lon CENTER_LON
                        Center longitude of the domain
  --hwidth_lat HWIDTH_LAT
                        Half-width of domain in latitude [degrees]
  --hwidth_lon HWIDTH_LON
                        Half-width of domain in longitude [degrees]
  --pole_lat POLE_LAT   Rotated pole latitude
  --pole_lon POLE_LON   Rotated pole longitude
  --ncells_boundary NCELLS_BOUNDARY
                        Lateral boundary cells to be removed (default: 0)
  --output OUTPUT       Output NetCDF file path
```

### Geographical lat/lon grid example
```
create-latlon-grid --grid_spacing 12.1 --center_lat 47.0 --center_lon 8.0 --hwidth_lat 12.0 --hwidth_lon 12.0 --ncells_boundary 16 --output latlon_grid.nc
```

#### Usage
```
usage: create-latlon-grid [-h] --grid_spacing GRID_SPACING --center_lat CENTER_LAT --center_lon CENTER_LON --hwidth_lat HWIDTH_LAT --hwidth_lon HWIDTH_LON
                          [--ncells_boundary NCELLS_BOUNDARY] --output OUTPUT

Generate a geographical lat/lon grid NetCDF file.

optional arguments:
  -h, --help            show this help message and exit
  --grid_spacing GRID_SPACING
                        Grid spacing in horizontal direction [km]
  --center_lat CENTER_LAT
                        Center latitude of the domain
  --center_lon CENTER_LON
                        Center longitude of the domain
  --hwidth_lat HWIDTH_LAT
                        Half-width of domain in latitude [degrees]
  --hwidth_lon HWIDTH_LON
                        Half-width of domain in longitude [degrees]
  --ncells_boundary NCELLS_BOUNDARY
                        Lateral boundary cells to be removed (default: 0)
  --output OUTPUT       Output NetCDF file path
```

## License

MIT