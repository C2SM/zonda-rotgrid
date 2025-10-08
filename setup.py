from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="zonda_rotgrid",
    version="0.2.0",
    description="Generate rotated coordinate grid NetCDF files for climate models based on Zonda input.",
    author="C2SM",
    license="MIT",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "xarray",
        "pyproj",
        "netCDF4",
        "h5netcdf",
        "scipy"
    ],
    entry_points={
        "console_scripts": [
            "create-rotated-grid=zonda_rotgrid.cli:main_rotated",
            "create-latlon-grid=zonda_rotgrid.cli:main_latlon"
        ]
    },
    python_requires=">=3.7",
    include_package_data=True,
    long_description=long_description,
    long_description_content_type="text/markdown",
)
