# zonda-rotgrid

Generate rotated coordinate grid NetCDF files for climate models based on [Zonda](https://zonda.ethz.ch/) input.



## Installation

```bash
pip install zonda-rotgrid
```

## Usage

After installation, use the command line tool:

```
create-rotated-grid --grid_spacing 12.1 --center_lat -0.77 --center_lon -5.11 --hwidth_lat 25.025 --hwidth_lon 24.365 --pole_lat 39.25 --pole_lon -162 --ncells_boundary 16 --output output.nc
```

Replace the arguments as needed for your domain or paste the command from Zonda directly.

## License

MIT