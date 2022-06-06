#!/bin/bash

set -exuo pipefail

THIS_DIR="$( cd "$(dirname "$0")"; pwd -P )"
export PYTHONPATH=$THIS_DIR/..

# Attempt to recreate exact Bootstrap output with this python code

# Create the NH sample file from scratch
python "${THIS_DIR}/../bt_py/compute_bt_ic.py"

# Compare with the output of the Fortran code
ofn_for="${THIS_DIR}/../SB2_NRT_programs/NH_20180217_SB2_NRT_f18.ic"
ofn_py="${THIS_DIR}/../bt_py/NH_20180217_py_NRT_f18.ic"

diff -s $ofn_for $ofn_py

# Use a python script to find out exactly where the differences are in outputs
if [[ -f "${THIS_DIR}/../bt_py/i2comp.py" ]]; then
  python "${THIS_DIR}/../bt_py/i2comp.py" $ofn_for $ofn_py
fi
