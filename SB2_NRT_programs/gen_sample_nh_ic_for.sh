#!/bin/bash

set -euxo pipefail

# Generate sample NH sea ice concentration field for a day (here Feb 17, 2018)

nh_bt_exe=./boot_driver_sb2_np_nrt
ymd_sample=20180217
sample_ic_fn=./NH_20180217_SB2_NRT_f18.ic
sample_ic_orig_fn=./NH_20180217_SB2_NRT_f18.ic.orig
sample_ic_orig_fn=./orig_output/NH_20180217_SB2_NRT_f18.ic.orig

# Remove prior versions of executable and output
rm -f ${nh_bt_exe}      # rm prior version, in case 'make' fails
rm -f ./NH_${ymd_sample}_SB2_NRT_f18.ic # remove previous output

# Make the the executable
make -f ./Makefile_sb2_np_nrt

# Create the ice conc output file
${nh_bt_exe} ${ymd_sample}

# Compare the ice conc output to the original
echo " "
diff -s ${sample_ic_orig_fn} ${sample_ic_fn}
echo " "
