import argparse
from .core import create_rotated_grid

def main():
    parser = argparse.ArgumentParser(description="Generate a rotated coordinate grid NetCDF file for climate models.")
    parser.add_argument("--grid_spacing", type=float, required=True, help="Grid spacing in horizontal direction [km]")
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
        grid_spacing=args.grid_spacing,
        center_lat=args.center_lat,
        center_lon=args.center_lon,
        hwidth_lat=args.hwidth_lat,
        hwidth_lon=args.hwidth_lon,
        pole_lat=args.pole_lat,
        pole_lon=args.pole_lon,
        ncells_boundary=args.ncells_boundary,
        output_path=args.output
    )
