import datetime as dt
from pathlib import Path
from typing import get_args
import numpy as np
from netCDF4 import Dataset
import rasterio

import click
from loguru import logger

from pm_icecon._types import Hemisphere
from pm_icecon.bt.api import a2l1c_goddard_bootstrap, amsr2_goddard_bootstrap
from pm_icecon.cli.util import datetime_to_date
from pm_icecon.fetch.au_si import AU_SI_RESOLUTIONS
from pm_icecon.util import standard_output_filename


def get_filename_attrs_a2l1c(fn):
    # Extract information from the filename
    # Assumes the base filename is in the form:
    #  bt_NH_20220101_a2l1c_6.25km.nc
    base_filename = fn.name.replace('.nc', '')
    alg, hem, ymd, tframe, tbsrc, res = base_filename.split('_')

    # Expecting only two possible grid types: EASE2 NH|SH at 6.25km
    try:
        assert alg == 'bt'
    except AssertionError:
        raise ValueError(f'Expecting alg "bt", but got: {alg}')

    try:
        assert hem in ('NH', 'SH')
    except AssertionError:
        raise ValueError(f'Expected hem of NH or SH, but got: {hem}')

    try:
        date = dt.datetime.strptime(ymd, '%Y%m%d').date()
    except ValueError as e:
        print(f'Could not determine date from filename {fn}')
        raise e

    try:
        assert tbsrc in ('a2l1c',)
    except AssertionError:
        raise ValueError(f'Expecting tbsrc of a2l1c, but got: {tbsrc}')

    try:
        assert res == '6.25km'
    except AssertionError:
        raise ValueError(f'Expecting resolution of 6.25km, but got: {res}')

    if hem == 'NH' and res == '6.25km':
        gridid = 'e2n6.25'
    elif hem == 'SH' and res == '6.25km':
        gridid = 'e2s6.25'
    else:
        raise ValueError(f'Could not determine gridid for hem {hem} and resolution {res}')  # noqa

    return date, gridid


def derive_geotiff_name_a2l1c(nc_fname):
    # Determine a geotiff name corresponding to the nc filename
    try:
        assert '.nc' in nc_fname.name
    except AssertionError:
        raise ValueError(f'expected .nc in filename: {nc_fname}')

    geotiff_dirname = nc_fname.parent
    geotiff_fname = nc_fname.name.replace('.nc', '.tif')

    geotiff_path = Path(geotiff_dirname, geotiff_fname)

    return geotiff_path


def update_netcdf_file(nc_path, file_date, file_gridid, ubyte_conc):
    # Replace input nc_path with new file
    # Note: Currently set up only to encode e2n6.25km partial grid
    if file_gridid == 'e2n6.25' and ubyte_conc.shape == (1680, 1680):
        # Info for Northern Hemisphere subset of EASE2 NH 6.25km grid
        xdim = 1680
        ydim = 1680
        res = 6250

        x_left = -xdim // 2 * res
        x_right = xdim // 2 * res
        y_top = ydim // 2 * res
        y_bottom = -ydim // 2 * res

        x_left_center = x_left + res // 2
        x_right_center = x_right - res // 2
        y_top_center = y_top - res // 2
        y_bottom_center = y_bottom + res // 2

    else:
        raise ValueError(f'This code only encodes EASE2 NH 6.25km subset')

    # Begin encoding georeferenced netCDF file
    ds = Dataset(nc_path, 'w')

    # Define the dimensions
    ydim, xdim = ubyte_conc.shape
    x = ds.createDimension('x', xdim)
    y = ds.createDimension('y', ydim)
    time = ds.createDimension('time', None)

    # Set global attributes
    ds.comment = f'Northern Hemisphere Bootstrap concentration for {file_date}'
    ds.Conventions = 'CF-1.10'

    # X values
    xs = ds.createVariable('x', np.float32, ('x',))
    xs_values = np.linspace(
        x_left_center,
        x_right_center,
        num=xdim,
        dtype=np.float32,
    )
    xs[:] = xs_values
    xs.standard_name = 'projection_x_coordinate'
    xs.long_name = 'x'
    xs.axis = 'X'
    xs.units = 'meters'
    xs.coverage_content_type = 'coordinate'
    xs.valid_range = np.array((x_left, x_right))

    # Y values
    ys = ds.createVariable('y', np.float32, ('y',))
    ys_values = np.linspace(
        y_top_center,
        y_bottom_center,
        num=ydim,
        dtype=np.float32,
    )
    ys[:] = ys_values
    ys.standard_name = 'projection_y_coordinate'
    ys.long_name = 'y'
    ys.axis = 'Y'
    ys.units = 'meters'
    ys.coverage_content_type = 'coordinate'
    ys.valid_range = np.array((y_bottom, y_top))

    reference_date = dt.date(1970, 1, 1)
    times = ds.createVariable('time', np.float64, ('time',))
    times.standard_name = 'time'
    times.long_name = 'ANSI date'
    times.calendar = 'standard'
    times.axis = 'T'
    times.units = f'days since {reference_date}'
    times.coverage_content_type = 'coordinate'
    times.valid_range = np.array((0, 30000))
    times[:] = (file_date - reference_date).days

    crs = ds.createVariable('crs', 'i4')
    crs.grid_mapping_name = 'lambert_azimuthal_equal_area'
    crs.longitude_of_projection = 0.
    crs.false_easting = 0.
    crs.false_northing = 0.
    crs.semi_major_axis = 6378137.
    crs.inverse_flattening = 298.257223563
    crs.GeoTransform = "-5250000 6250 0 5250000 0 -6250"
    crs.long_name = "NSIDC_EASE2_N6.25km_subset"
    crs.latitude_of_projection_origin = 90.
    crs.proj4text = "+proj=laea +lat_0=90 +lon_0=0 +x_0=0 +y_0=0 +ellps=WGS84 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs"  # noqa
    crs.srid = "urn:ogc:def:crs:EPSG::6931"
    crs.crs_wkt = "PROJCS[\"unnamed\",GEOGCS[\"WGS 84\",DATUM[\"WGS_1984\",SPHEROID[\"WGS 84\",6378137,298.257223563,AUTHORITY[\"EPSG\",\"7030\"]],TOWGS84[0,0,0,0,0,0,0],AUTHORITY[\"EPSG\",\"6326\"]],PRIMEM[\"Greenwich\",0,AUTHORITY[\"EPSG\",\"8901\"]],UNIT[\"degree\",0.0174532925199433,AUTHORITY[\"EPSG\",\"9108\"]],AUTHORITY[\"EPSG\",\"4326\"]],PROJECTION[\"Lambert_Azimuthal_Equal_Area\"],PARAMETER[\"latitude_of_center\",90],PARAMETER[\"longitude_of_center\",0],PARAMETER[\"false_easting\",0],PARAMETER[\"false_northing\",0],UNIT[\"Meter\",1],AUTHORITY[\"epsg\",\"6931\"]]atitude_of_center\",90],PARAMETER[\"longitude_of_center\",0],PARAMETER[\"false_easting\",0],PARAMETER[\"false_northing\",0],UNIT[\"Meter\",1],AUTHORITY[\"epsg\",\"6931\"]]"  # noqa

    conc = ds.createVariable('conc', np.uint8, ('time', 'y', 'x'), zlib=True)  # noqa
    conc[0, :, :] = ubyte_conc[:, :]
    conc.coverage_content_type = 'image'

    conc.coordinates = 'time y x'
    conc.grid_mapping = 'crs'

    conc.units = '1'
    conc.long_name = 'Sea Ice Concentration'
    conc.standard_name = 'sea_ice_area_fraction'
    conc.valid_range = np.array((0, 100))
    conc.flag_values = np.array((0,))
    conc.flag_meanings = 'non_ocean_pixel'
    conc.packing_convention = 'netCDF'
    conc.packing_convention_description = 'unpacked = scale_factor * packed + add_offset'
    conc.scale_factor = 0.01
    conc.add_offset = 0.

    ds.close()
    logger.info(f'Re-encoded {nc_path}')


def add_info_to_netcdf_file_a2l1c(nc_path):
    # Read in siconc data from a bare-bones netCDF file
    # Create a new netCDF file of the same name with georeference info
    # Write out the new data to the same input file
    file_date, file_gridid = get_filename_attrs_a2l1c(nc_path)

    ds = Dataset(nc_path)
    ds_conc = np.array(ds.variables['conc'])
    ds.close()

    rounded_conc = np.round(ds_conc)
    try:
        assert rounded_conc.min() >= 0
        assert rounded_conc.max() <= 255
    except AssertionError:
        raise ValueError(f'Concentration outside expect bounds of 0 ({rounded_conc.min()}) and 255 ({rounded_conc.max()})')  # noqa
    ubyte_conc = rounded_conc.astype(np.uint8)

    update_netcdf_file(nc_path, file_date, file_gridid, ubyte_conc)


def get_geotiff_colormap():
    # Define a colormap for geotiff files

    colormap_dict = {
        0: (0, 0, 100),  # Blue for no-ice ocean

        1: (2, 2, 2),  # Gray value for concentration is double the conc
        2: (4, 4, 4),
        3: (6, 6, 6),
        4: (8, 8, 8),
        5: (10, 10, 10),
        6: (12, 12, 12),
        7: (14, 14, 14),
        8: (16, 16, 16),
        9: (18, 18, 18),
        10: (20, 20, 20),
        11: (22, 22, 22),
        12: (24, 24, 24),
        13: (26, 26, 26),
        14: (28, 28, 28),
        15: (30, 30, 30),
        16: (32, 32, 32),
        17: (34, 34, 34),
        18: (36, 36, 36),
        19: (38, 38, 38),
        20: (40, 40, 40),
        21: (42, 42, 42),
        22: (44, 44, 44),
        23: (46, 46, 46),
        24: (48, 48, 48),
        25: (50, 50, 50),
        26: (52, 52, 52),
        27: (54, 54, 54),
        28: (56, 56, 56),
        29: (58, 58, 58),
        30: (60, 60, 60),
        31: (62, 62, 62),
        32: (64, 64, 64),
        33: (66, 66, 66),
        34: (68, 68, 68),
        35: (70, 70, 70),
        36: (72, 72, 72),
        37: (74, 74, 74),
        38: (76, 76, 76),
        39: (78, 78, 78),
        40: (80, 80, 80),
        41: (82, 82, 82),
        42: (84, 84, 84),
        43: (86, 86, 86),
        44: (88, 88, 88),
        45: (90, 90, 90),
        46: (92, 92, 92),
        47: (94, 94, 94),
        48: (96, 96, 96),
        49: (98, 98, 98),
        50: (100, 100, 100),
        51: (102, 102, 102),
        52: (104, 104, 104),
        53: (106, 106, 106),
        54: (108, 108, 108),
        55: (110, 110, 110),
        56: (112, 112, 112),
        57: (114, 114, 114),
        58: (116, 116, 116),
        59: (118, 118, 118),
        60: (120, 120, 120),
        61: (122, 122, 122),
        62: (124, 124, 124),
        63: (126, 126, 126),
        64: (128, 128, 128),
        65: (130, 130, 130),
        66: (132, 132, 132),
        67: (134, 134, 134),
        68: (136, 136, 136),
        69: (138, 138, 138),
        70: (140, 140, 140),
        71: (142, 142, 142),
        72: (144, 144, 144),
        73: (146, 146, 146),
        74: (148, 148, 148),
        75: (150, 150, 150),
        76: (152, 152, 152),
        77: (154, 154, 154),
        78: (156, 156, 156),
        79: (158, 158, 158),
        80: (160, 160, 160),
        81: (162, 162, 162),
        82: (164, 164, 164),
        83: (166, 166, 166),
        84: (168, 168, 168),
        85: (170, 170, 170),
        86: (172, 172, 172),
        87: (174, 174, 174),
        88: (176, 176, 176),
        89: (178, 178, 178),
        90: (180, 180, 180),
        91: (182, 182, 182),
        92: (184, 184, 184),
        93: (186, 186, 186),
        94: (188, 188, 188),
        95: (190, 190, 190),
        96: (192, 192, 192),
        97: (194, 194, 194),
        98: (196, 196, 196),
        99: (198, 198, 198),
        100: (200, 200, 200),

        254: (10, 120, 10),  # Green for land

        255: (250, 0, 0),    # Red for missing
    }

    return colormap_dict


def create_equivalent_geotiff_a2l1c(nc_path, geotiff_path):
    # Create equivalent geotiff from netCDF file

    nc_ds = Dataset(nc_path, 'r')
    nc_ds.set_auto_maskandscale(False)
    conc_data = np.array(nc_ds['conc']).squeeze()

    geotiff_metadata = {
        'driver': 'GTiff',
        'interleave': 'band',
        'tiled': False,
        'count': 1,
        'nodata': 255,
        'dtype': 'uint8',
        'width': 1680,
        'height': 1680,
        'transform': rasterio.transform.Affine.from_gdal(
            -5250000, 6250, 0, 5250000, 0, -6250)
    }

    if 'EASE2_N' in nc_ds['crs'].long_name:
        geotiff_metadata['crs'] = rasterio.crs.CRS.from_epsg(6931)
    elif 'EASE2_S' in nc_ds['crs'].long_name:
        geotiff_metadata['crs'] = rasterio.crs.CRS.from_epsg(6932)
    else:
        raise ValueError(f'Could not determine epsg code from crs.long_name: {nc_ds["crs"].long_name}')  # noqa

    geotiff_colormap = get_geotiff_colormap()

    with rasterio.open(
            geotiff_path,
            'w',
            **geotiff_metadata,
            COMPRESS='DEFLATE') as dst:
        dst.write(conc_data[:, :], indexes=1)
        dst.write_colormap(1, geotiff_colormap)

    logger.info(f'Created geotiff file: {geotiff_path}')


# Click definitions for "amsr2" which uses AU25
@click.command()
@click.option(
    '-d',
    '--date',
    required=True,
    type=click.DateTime(formats=('%Y-%m-%d',)),
    callback=datetime_to_date,
)
@click.option(
    '-h',
    '--hemisphere',
    required=True,
    type=click.Choice(get_args(Hemisphere)),
)
@click.option(
    '-o',
    '--output-dir',
    required=True,
    type=click.Path(
        exists=True,
        file_okay=False,
        dir_okay=True,
        writable=True,
        resolve_path=True,
        path_type=Path,
    ),
)
@click.option(
    '-r',
    '--resolution',
    required=True,
    type=click.Choice(get_args(AU_SI_RESOLUTIONS)),
)
def amsr2(
    *,
    date: dt.date,
    hemisphere: Hemisphere,
    output_dir: Path,
    resolution: AU_SI_RESOLUTIONS,
):
    """Run the bootstrap algorithm with ASMR2 data.

    AMSRU2 brightness temperatures are fetched from AU_SI25.

    The resulting concentration field is saved to a netcdf file in the given
    `output_dir` with the form `{N|S}H_{YYYYMMDD}_py_NRT_amsr2.nc`
    """
    conc_ds = amsr2_goddard_bootstrap(
        date=date,
        hemisphere=hemisphere,
        resolution=resolution,
    )

    output_fn = standard_output_filename(
        hemisphere=hemisphere,
        date=date,
        sat='u2',
        resolution=f'{resolution}km',
        algorithm='bt',
    )
    output_path = output_dir / output_fn
    conc_ds.to_netcdf(output_path)
    logger.info(f'Wrote AMSR2 concentration field: {output_path}')


# Click definitions for 'a2l1c' which uses 6.25km fields derived from 0763
@click.command()
@click.option(
    '-d',
    '--date',
    required=True,
    type=click.DateTime(formats=('%Y-%m-%d',)),
    callback=datetime_to_date,
)
@click.option(
    '-h',
    '--hemisphere',
    required=True,
    type=click.Choice(get_args(Hemisphere)),
)
@click.option(
    '-o',
    '--output-dir',
    required=True,
    type=click.Path(
        exists=True,
        file_okay=False,
        dir_okay=True,
        writable=True,
        resolve_path=True,
        path_type=Path,
    ),
)
@click.option(
    '-t',
    '--tb_dir',
    required=True,
    type=click.Path(
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
        path_type=Path,
    ),
)
@click.option(
    '-a',
    '--anc_dir',
    required=True,
    type=click.Path(
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
        path_type=Path,
    ),
)
@click.option(
    '-p',
    '--nctbfn_template',
    required=False,
    default="NSIDC-0763-EASE2_{hemlet}{gridres}km-GCOMW1_AMSR2-{year}{doy}-{capchan}-{tim}-SIR-PPS_XCAL-v1.1.nc",
)
@click.option(
    '-f',
    '--timeframe',
    required=False,
    default="M",
)
def a2l1c(
    *,
    date: dt.date,
    hemisphere: Hemisphere,
    output_dir: Path,
    tb_dir: Path,
    anc_dir: Path,
    nctbfn_template: str,
    timeframe: str,
):

    """Run the bootstrap algorithm with 'a2l1c' data.

    AMSR brightness temperatures are fetched from 6.25km raw data fields
    derived from nsidc0763 (unpublished) files created from L1C fields

    The resulting concentration field is saved to a netcdf file in the given
    `output_dir`.
    """
    resolution = '6.25km'
    conc_ds = a2l1c_goddard_bootstrap(
        date=date,
        hemisphere=hemisphere,
        tb_dir=tb_dir,
        anc_dir=anc_dir,
        ncfn_=nctbfn_template,
        timeframe=timeframe,
    )

    output_fn = standard_output_filename(
        hemisphere=hemisphere,
        date=date,
        sat='a2l1c',
        resolution=resolution,
        algorithm='bt',
        timeframe=timeframe,
    )
    output_path = output_dir / output_fn
    # conc_ds.to_netcdf(output_path)
    conc_ds.astype(np.float32).to_netcdf(
        output_path, encoding={'conc': {'zlib': True}})

    logger.info(f'Wrote a2l1c concentration field: {output_path}')

    # Note: the following command replaces the output file
    add_info_to_netcdf_file_a2l1c(output_path)
    logger.info(f'Re-wrote AMSR2 concentration netCDF file: {output_path}')

    # Write an equivalent geotiff file
    geotiff_output_path = derive_geotiff_name_a2l1c(output_path)
    create_equivalent_geotiff_a2l1c(output_path, geotiff_output_path)
    logger.info(f'Wrote AMSR2 concentration geotiff: {geotiff_output_path}')


@click.group(name='bootstrap')
def cli():
    """Run the bootstrap algorithm."""
    ...


cli.add_command(amsr2)
cli.add_command(a2l1c)


if __name__ == '__main__':
    cli()
