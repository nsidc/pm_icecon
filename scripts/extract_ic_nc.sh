#!/bin/bash

# extract_ic_nc.sh

# Extract the sea ice conc field from the netCDF output of our AMSR2 Bootstrap code

ifn="$1"
dummyfn=dummy.nc

if [ -z "$ifn" ]; then
  echo "Usage:\n  ./extract_ic_nc.sh <ncfn>"
  exit
else
  bfn=$(basename $ifn)
  bfn=${bfn%.*}
  ofn=${bfn}_conc.dat
  ncks -C -O -v conc -b $ofn $ifn $dummyfn
  rm $dummyfn
fi

echo "Wrote: $ofn"
