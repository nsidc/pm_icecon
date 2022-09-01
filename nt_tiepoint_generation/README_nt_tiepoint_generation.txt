=================================
README_nt_tiepoint_generation.txt
=================================

Three tiepoints are required for the NASA Team sea ice concentration algorithm
  - Open Water
  - First Year Ice
  - Multi Year Ice
and for three channels:
  - 19H
  - 19V
  - 37V

Traditionally, this is done by matching the tie points for a calibrated 
sensor with the new sensor.

Here, we compute tie points for AMSR2 by computing a linear regression for
the AMSR2 brightness temperatures (Tbs) relative to the RSS F17 Tbs.

In order to best match the spatial resolutions of the gridded Tbs, the F17
Tbs are taken from the gridded F17 fields of NSIDC-0001 and the 10km-resolution
gridded AMSR2 Tbs from AMSRU.

We minimize the number of low-quality Tb observations in the linear regression
domain by excluding ocean pixels within three pixels of the land mask, and
by using day-of-year valid ice mask.

The expanded land mask is created by applying a kernel of shape:
  0 0 1 1 1 0 0
  0 1 1 1 1 1 0
  1 1 1 1 1 1 1
  1 1 1 1 1 1 1
  1 1 1 1 1 1 1
  0 1 1 1 1 1 0
  0 0 1 1 1 0 0
...to each pixel of the land mask.  The routines that calculate these
expanded land masks are:
  - compute_expanded_landmask_psn25.py  <-- for the Northern Hemisphere
  - compute_expanded_landmask_pss25.py  <-- for the Southern Hemisphere
for the Northern and Southern hemisphere, respectively.

The files used for the landmask here are:
  - psn25_landmask.dat  <-- for the Northern Hemisphere
  - pss25_landmask.dat  <-- for the Southern Hemisphere

The day-of-year valid ice mask was computed by looping through all NSIDC-0079
(Bootstrap) sea ice concentration fields created from F13 or F17 inputs.
These go back to May 10, 1995 and run through Dec 31, 2021.  Day-of-year 365 and 366 are set equal to each other.  The NH and SH daily masks were created
using:
  - make_doy_raw_icemasks.py

After these daily valid ice masks are created, a new valid ice mask is created
by including adjacent days in the valid expanded ice mask.  E.g., the valid
ice mask for day-of-year 32 includes all locations where NSIDC-0079 had sea
ice > 15% for Jan 31, Feb 1, and/or Feb 2.  These temporally-expanded 
day-of-year valid ice masks were created by running:
  - create_doymasks.py

With the Tb domain now established -- temporally with the 3-day day-of-year valid ice mask, and spatially with the 3-pixel-from-coast expanded land mask -- we
compute a linear regression of the AMSR2 Tbs with the F17 Tbs for each day
of 2021 using:
  - compute_tb_linreg_psn25.py  <-- for the Northern Hemisphere
  - compute_tb_linreg_pss25.py  <-- for the Southern Hemisphere

This yields a slope and a y-intercept for each day of 2021.  These slopes and
intercepts are then averaged to find a mean slope and intercept for each
channel.  Finally, this mean slope and mean intercept is used to calculate
the equivalent tie points for AMSR2 by applying the regression coefficients
to the F17 tie points using the scripts:
  - calc_nt_tiepoints_psn25.py  <-- for the Northern Hemisphere
  - calc_nt_tiepoints_pss25.py  <-- for the Southern Hemisphere

The bash scripts for computing these sets of tie points are
  - determine_nt_tiepoints_psn25.sh  <-- for the Northern Hemisphere
  - determine_nt_tiepoints_pss25.sh  <-- for the Southern Hemisphere

