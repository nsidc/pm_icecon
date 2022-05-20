#!/bin/bash

# gen_sample_nt_sh_C.sh

HOME=/home/scotts/sic_py/nt_orig
SPATIALINT_DIR=$HOME/system/new_1_tbs_spatially_filtered
TB_HOME=/home/scotts/sic_py/nt_orig/system/0_tbs/

### Spatial interpolation ### ----------------------------------------

# set -A channels 19h 19v 22v 37h 37v 91h 91v

# Spatially interpolate 19h
echo "1
/home/scotts/sic_py/nt_orig/system/0_tbs/SSSSS1DTB01/sssss1d17tb19h2018001" > tmp_filelist.txt

/home/scotts/sic_py/nt_orig/bin/SpatialInt_sp << args
19h
tmp_filelist.txt
$SPATIALINT_DIR
args

# Spatially interpolate 19v
echo "1
/home/scotts/sic_py/nt_orig/system/0_tbs/SSSSS1DTB01/sssss1d17tb19v2018001" > tmp_filelist.txt

/home/scotts/sic_py/nt_orig/bin/SpatialInt_sp << args
19v
tmp_filelist.txt
$SPATIALINT_DIR
args

# Spatially interpolate 22v
echo "1
/home/scotts/sic_py/nt_orig/system/0_tbs/SSSSS1DTB01/sssss1d17tb22v2018001" > tmp_filelist.txt

/home/scotts/sic_py/nt_orig/bin/SpatialInt_sp << args
22v
tmp_filelist.txt
$SPATIALINT_DIR
args

# Spatially interpolate 37h
echo "1
/home/scotts/sic_py/nt_orig/system/0_tbs/SSSSS1DTB01/sssss1d17tb37h2018001" > tmp_filelist.txt

/home/scotts/sic_py/nt_orig/bin/SpatialInt_sp << args
37h
tmp_filelist.txt
$SPATIALINT_DIR
args

# Spatially interpolate 37v
echo "1
/home/scotts/sic_py/nt_orig/system/0_tbs/SSSSS1DTB01/sssss1d17tb37v2018001" > tmp_filelist.txt

/home/scotts/sic_py/nt_orig/bin/SpatialInt_sp << args
37v
tmp_filelist.txt
$SPATIALINT_DIR
args

### Run the Concentration algorithm
conc_dir=/home/scotts/sic_py/nt_orig/system/new_2_iceconcentrations
conc_exe=/home/scotts/sic_py/nt_orig/bin/seaice5con
sst_dir=/home/scotts/sic_py/nt_orig/system/new_3_icecons_spill_sst
sst_exe=/home/scotts/sic_py/nt_orig/bin/apply_sst_s

# Create sssss1d17tcon2018001 in conc_dir
cd $conc_dir
${conc_exe} 001 2018 001 2018 TOT_CON ssmif17 s
# ${sst_exe} < ${conc_dir}/sssss1d17tcon2018001 
${sst_exe} << args
${conc_dir}/sssss1d17tcon2018001 
args
mv ${conc_dir}/sssss1d17tcon2018001.spill_sst ${sst_dir}
