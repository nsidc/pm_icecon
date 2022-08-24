#!/bin/bash

# determine_nt_tiepoints_pss25.sh

# Script to run the codes that determine the NT tiepoints by
# matching F17 and AMSR2 values

# Create a mask that excludes grid cells close to the coast
# input: ./pss25_landmask.dat
# output: ./pss25_expanded_landmask.dat
expland_fn=./pss25_expanded_landmask.dat
if [ ! -f $expland_fn ]; then
  python compute_expanded_landmask_pss25.py
else
  echo "Using existing expanded land mask: $expland_fn"
fi

# Create one-day ever-valid ice masks, both NH and SH
# input: 0079 files
# output: ./bt_doymasks_raw/ files, both NH and SH
raw_doymask_dir=./bt_doymasks_raw
if [ ! -d $raw_doymask_dir ]; then
  python ./make_doy_raw_icemask_pss25.py
else
  echo "Using existing raw doymasks in: $raw_doymask_dir"
fi

# Use the one-day masks to create 3-day masks
sh_doydir=./bt_doymasks_sh
if [ ! -d $sh_doydir ]; then
  python ./create_doymask_pss25.py
else
  echo "Using doymasks in existing sh doymask dir: $sh_doydir"
fi

# Compute the daily linear regressions
daily_lr_fn=f17_vs_amsr2_tb_regressions_2021_sh.txt
if [ ! -f $daily_lr_fn ]; then
  python ./compute_tb_linreg_pss25.py |& tee ${daily_lr_fn}
else
  echo "Using existing daily linear regressions from: $daily_lr_fn"
fi

# Compute the SH tie points
sh_tiepoint_fn=nt_tiepoints_for_amsr2_sh.txt
echo "Writing SH tiepoints to: $sh_tiepoint_fn"
python calc_nt_tiepoints_pss25.py ${daily_lr_fn} |& tee ${sh_tiepoint_fn}

echo "Wrote: $sh_tiepoint_fn"
