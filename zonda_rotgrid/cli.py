import argparse
from .core import create_rotated_grid, create_latlon_grid

def main_rotated():
	parser = argparse.ArgumentParser(description="Generate a rotated coordinate grid NetCDF file.")
	group = parser.add_mutually_exclusive_group(required=True)
	group.add_argument("--grid_spacing", type=float, help="Grid spacing in horizontal direction [km]")
	group.add_argument("--dlon", type=float, help="Grid spacing in longitude direction [degrees]")
	parser.add_argument("--dlat", type=float, help="Grid spacing in latitude direction [degrees] (required if --dlon is used)")
	parser.add_argument("--center_lat", type=float, required=True, help="Center latitude of the domain")
	parser.add_argument("--center_lon", type=float, required=True, help="Center longitude of the domain")
	parser.add_argument("--hwidth_lat", type=float, required=True, help="Half-width of domain in latitude [degrees]")
	parser.add_argument("--hwidth_lon", type=float, required=True, help="Half-width of domain in longitude [degrees]")
	parser.add_argument("--pole_lat", type=float, required=True, help="Rotated pole latitude")
	parser.add_argument("--pole_lon", type=float, required=True, help="Rotated pole longitude")
	parser.add_argument("--ncells_boundary", type=int, default=0, help="Lateral boundary cells to be removed (default: 0)")
	parser.add_argument("--output", type=str, required=True, help="Output NetCDF file path")

	args = parser.parse_args()
	# If dlon is used, dlat must also be provided
	if args.dlon is not None:
		if args.dlat is None:
			parser.error("--dlat must be provided when using --dlon.")
		create_rotated_grid(
			dlat=args.dlat,
			dlon=args.dlon,
			center_lat=args.center_lat,
			center_lon=args.center_lon,
			hwidth_lat=args.hwidth_lat,
			hwidth_lon=args.hwidth_lon,
			pole_lat=args.pole_lat,
			pole_lon=args.pole_lon,
			ncells_boundary=args.ncells_boundary,
			output_path=args.output
		)
	else:
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

def main_latlon():
	parser = argparse.ArgumentParser(description="Generate a geographical lat/lon grid NetCDF file.")
	parser.add_argument("--grid_spacing", type=float, required=True, help="Grid spacing in horizontal direction [km]")
	parser.add_argument("--center_lat", type=float, required=True, help="Center latitude of the domain")
	parser.add_argument("--center_lon", type=float, required=True, help="Center longitude of the domain")
	parser.add_argument("--hwidth_lat", type=float, required=True, help="Half-width of domain in latitude [degrees]")
	parser.add_argument("--hwidth_lon", type=float, required=True, help="Half-width of domain in longitude [degrees]")
	parser.add_argument("--ncells_boundary", type=int, default=0, help="Lateral boundary cells to be removed (default: 0)")
	parser.add_argument("--output", type=str, required=True, help="Output NetCDF file path")

	args = parser.parse_args()
	create_latlon_grid(
		grid_spacing=args.grid_spacing,
		center_lat=args.center_lat,
		center_lon=args.center_lon,
		hwidth_lat=args.hwidth_lat,
		hwidth_lon=args.hwidth_lon,
		ncells_boundary=args.ncells_boundary,
		output_path=args.output
	)
