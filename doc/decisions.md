# Decisions

Documentation about decisions made about the bootstrap and nasateam algorithms.

# Bootstrap

## 6 GHz weather filter

The code we received from Goddard for processing AMSR 2 data includes a 6GHz
vertical channel weather filter.

We have decided NOT to use the 6GHz channel as an additional wather filter for
ASMSR 2 data so as to be consistent across the entire timeseries, including when
using sensors that lack a 6GHz vertical channel.

The Goddard weather filter code that uses `wintrc2` and `wslope2` is located in
`/share/apps/amsr2-cdr/cdr_testdata/bt_amsru_regression/orig_goddard_bt_code/ret_water_amsru2.f`. Originally
set in ret_parameters_amsru2.f.

