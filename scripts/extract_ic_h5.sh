#!/bin/bash

# extract_ic_h5.sh

# Extract the sea ice conc field from the hdf5 AMSRU2 file

ifn="$1"

if [ -z "$ifn" ]; then
  echo "Usage:\n  ./extract_ic_h5.sh <h5fn>"
  exit
else
  bfn=$(basename $ifn)
  bfn=${bfn%.*}
  ofn=${bfn}_conc.dat

  # Remove the output file if it exists,
  # so we can tell if something goes wrong
  if [ -e $ofn ]; then
    rm $ofn
  fi

  h5dump \
    -d "HDFEOS/GRIDS/NpPolarGrid25km/Data Fields/SI_25km_NH_ICECON_DAY" \
    -b LE \
    -o $ofn \
    $ifn
fi

if [ -e $ofn ]; then
  echo "Wrote: $ofn"
else
  echo "Hmmm.  No such output file(!): $ofn"
fi
