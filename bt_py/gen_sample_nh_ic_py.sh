#!/bin/bash

# gen_sample_nh_ic_py.sh

# Attempt to recreate exact Bootstrap output with this python code

# Create the NH sample file from scratch
python compute_bt_ic.py

# Compare with the output of the Fortran code
ofn_for=../SB2_NRT_programs/NH_20180217_SB2_NRT_f18.ic
ofn_py=./NH_20180217_py_NRT_f18.ic

diff -s $ofn_for $ofn_py

# Use a python script to find out exactly where the differences are in outputs
if [[ -f ./i2comp.py ]]; then
  python i2comp.py $ofn_for $ofn_py
fi
