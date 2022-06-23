import datetime as dt
from pathlib import Path

import xarray as xr

import cdr_amsr2.bt.compute_bt_ic as bt
from cdr_amsr2.constants import PACKAGE_DIR
from cdr_amsr2.fetch.au_si25 import get_au_si25_tbs

if __name__ == '__main__':
    date = dt.date(2020, 1, 1)
    xr_tbs = get_au_si25_tbs(
        base_dir=Path('/ecs/DP1/AMSA/AU_SI25.001/'),
        date=date,
        hemisphere='north',
    )

    # params = bt.import_cfg_file(PACKAGE_DIR / 'bt' / 'ret_ic_params.json')
    params = bt.import_cfg_file(PACKAGE_DIR / 'bt' / 'ret_ic_params_amsru.json')
    variables = bt.import_cfg_file(PACKAGE_DIR / 'bt' / 'ret_ic_variables_amsru.json')

    tbs = {
        'v19': xr_tbs['v18'].data,
        'v37': xr_tbs['v36'].data,
        'h37': xr_tbs['h36'].data,
        'v22': xr_tbs['v23'].data,
    }

    land_arr = bt._get_land_arr(params)

    tb_mask = bt.tb_data_mask(
        tbs=(
            tbs['v37'],
            tbs['h37'],
            tbs['v19'],
            tbs['v22'],
        ),
        min_tb=params['mintb'],
        max_tb=params['maxtb'],
    )

    tbs = bt.xfer_tbs_nrt(tbs['v37'], tbs['h37'], tbs['v19'], tbs['v22'], params['sat'])

    # params.update(bt.ret_para_nsb2('vh37', params['sat'], params['seas']))
    para_vals_vh37 = bt.ret_para_nsb2('vh37', params['sat'], params['seas'])
    params['wintrc'] = para_vals_vh37['wintrc']
    params['wslope'] = para_vals_vh37['wslope']
    params['wxlimt'] = para_vals_vh37['wxlimt']
    params['ln1'] = para_vals_vh37['lnline']
    params['lnchk'] = para_vals_vh37['lnchk']
    variables['wtp'] = para_vals_vh37['wtp']
    variables['itp'] = para_vals_vh37['itp']

    water_arr = bt.ret_water_ssmi(
        tbs['v37'],
        tbs['h37'],
        tbs['v22'],
        tbs['v19'],
        land_arr,
        tb_mask,
        params['wslope'],
        params['wintrc'],
        params['wxlimt'],
        params['ln1'],
    )

    # Set wtp, which is tp37v and tp37h
    variables['wtp37v'] = bt.ret_wtp_32(water_arr, tbs['v37'])
    variables['wtp37h'] = bt.ret_wtp_32(water_arr, tbs['h37'])

    if (variables['wtp'][0] - 10) < variables['wtp37v'] < (variables['wtp'][0] + 10):
        variables['wtp'][0] = variables['wtp37v']
    if (variables['wtp'][1] - 10) < variables['wtp37h'] < (variables['wtp'][1] + 10):
        variables['wtp'][1] = variables['wtp37h']

    calc_vh37 = bt.ret_linfit_32(
        land_arr,
        tb_mask,
        tbs['v37'],
        tbs['h37'],
        params['ln1'],
        params['lnchk'],
        params['add1'],
        water_arr,
    )
    variables['vh37'] = calc_vh37

    variables['adoff'] = bt.ret_adj_adoff(variables['wtp'], variables['vh37'])

    # para_vals = bt.ret_para_nsb2('v1937', params['sat'], params['seas'])
    para_vals_v1937 = bt.ret_para_nsb2('v1937', params['sat'], params['seas'])
    params['ln2'] = para_vals_v1937['lnline']
    variables['wtp2'] = para_vals_v1937['wtp']
    variables['itp2'] = para_vals_v1937['itp']
    variables['v1937'] = para_vals_v1937['iceline']

    variables['wtp19v'] = bt.ret_wtp_32(water_arr, tbs['v19'])

    if (variables['wtp2'][0] - 10) < variables['wtp37v'] < (variables['wtp2'][0] + 10):
        variables['wtp2'][0] = variables['wtp37v']
    if (variables['wtp2'][1] - 10) < variables['wtp19v'] < (variables['wtp2'][1] + 10):
        variables['wtp2'][1] = variables['wtp19v']

    # Try the ret_para... values for v1937
    calc_v1937 = bt.ret_linfit_32(
        land_arr,
        tb_mask,
        tbs['v37'],
        tbs['v19'],
        params['ln2'],
        params['lnchk'],
        params['add2'],
        water_arr,
        tba=tbs['h37'],
        iceline=variables['vh37'],
        adoff=variables['adoff'],
    )
    variables['v1937'] = calc_v1937

    # ## LINES calculating radslp1 ... to radlen2 ###
    variables = bt.calc_rad_coeffs_32(variables)

    # ## LINES with loop calling (in part) ret_ic() ###
    iceout = bt.calc_bt_ice(params, variables, tbs, land_arr, water_arr, tb_mask)

    # *** Do sst cleaning ***
    iceout_sst = bt.sst_clean_sb2(
        iceout, params['missval'], params['landval'], params['month']
    )

    # *** Do spatial interp ***
    iceout_sst = bt.spatial_interp(
        iceout_sst,
        params['missval'],
        params['landval'],
        (
            PACKAGE_DIR / '../legacy/SB2_NRT_programs' / params['raw_fns']['nphole']
        ).resolve(),
    )

    # *** Do spatial interp ***
    iceout_fix = bt.coastal_fix(
        iceout_sst, params['missval'], params['landval'], params['minic']
    )
    iceout_fix[iceout_fix < params['minic']] = 0

    # *** Do fix_output ***
    fixout = bt.fix_output_gdprod(
        iceout_fix,
        params['minval'],
        params['maxval'],
        params['landval'],
        params['missval'],
    )

    # cast to xr ds to make writing to nc easy!
    ofn = f'NH_{date:%Y%m%d}_py_NRT_amsr2.nc'
    output_path = (PACKAGE_DIR / '..' / ofn).resolve()
    ds = xr.Dataset({'conc': (('y', 'x'), fixout)})
    ds.to_netcdf(output_path)
    print(f'Wrote output file: {output_path}')

    print('\nparams:')
    for key in params.keys():
        print(f'  {key}: {params[key]}')

    print('\nvariables:')
    for key in variables.keys():
        print(f'  {key}: {variables[key]}')
