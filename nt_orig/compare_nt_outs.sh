#!/bin/bash

# compare_nt_outs.sh

# verify that outputs are the same between Goddard fields and those
# produced here

for fdate in {2018001..2018010}; do 
  # NH
  fn1=/data/NT_fromGoddard/3_icecons_spill_sst/nssss1d17tcon${fdate}.spill_sst
  fn2=./system/3_icecons_spill_sst/nssss1d17tcon${fdate}.spill_sst

  echo "NH: $fdate"
  cmp -i 300 $fn1 $fn2

  #SH
  fn1=/data/NT_fromGoddard/3_icecons_spill_sst/sssss1d17tcon${fdate}.spill_sst
  fn2=./system/3_icecons_spill_sst/sssss1d17tcon${fdate}.spill_sst
  echo "SH: $fdate"
  cmp -i 300 $fn1 $fn2

  echo " "
done
