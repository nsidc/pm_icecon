#!/bin/bash

# extract_rawdat.sh

# Extract the raw 'conc' field from the given netCDF file

ifn=${1}
if [ -z ${ifn} ]; then
  echo "Usage: $0 <nc_filename>"
  echo "eg:"
  echo "  $0 ./bt_out/NH_20220608_py_NRT_amsr2.nc"
  exit 1
fi

indir=$(dirname $ifn)
bfn=$(basename $ifn)

# echo "indir: ${indir}"
# echo "  bfn: ${bfn}"

# replace ".nc" in the input filename with "_conc.dat"
ofn=${ifn%.nc}_conc.dat

ncdummy=.dummy.nc
ncks -C -O -v conc -b ${ofn} ${ifn} ${ncdummy}
rm ${ncdummy}

echo "Extracted 'conc' field from: ${ifn}"
echo "      to raw binary fileofn: ${ofn}"
