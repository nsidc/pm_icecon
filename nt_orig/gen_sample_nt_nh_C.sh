#!/bin/bash

# gen_sample_nt_nh_C.sh

# *Very* rough translation of .ksh script for a single file to use
# for comparison purposes

HOME=/home/scotts/sic_py/nt_orig

src_dir=${HOME}/C_CODE

SPATIALINT_DIR=$HOME/system/new_1_tbs_spatially_filtered
TB_HOME=/home/scotts/sic_py/nt_orig/system/0_tbs/

spi_exe=/home/scotts/sic_py/nt_orig/bin/SpatialInt_np
spi_tmpfile=tmp_filelist.txt

conc_dir=/home/scotts/sic_py/nt_orig/system/new_2_iceconcentrations
conc_exe=/home/scotts/sic_py/nt_orig/bin/seaice5con

sst_dir=/home/scotts/sic_py/nt_orig/system/new_3_icecons_spill_sst
sst_exe=/home/scotts/sic_py/nt_orig/bin/apply_sst_n
oldfile=system/3_icecons_spill_sst/nssss1d17tcon2018001.spill_sst
newfile=system/new_3_icecons_spill_sst/nssss1d17tcon2018001.spill_sst

# Clean up before starting
rm -fv ${spi_exe}
rm -fv ${spi_tmpfile}
rm -fv ${conc_exe}
rm -fv ${sst_exe}
rm -fv ${newfile}

# make the files
cd ${src_dir}
./makeall.csh

### Spatial interpolation ### ----------------------------------------
cd $HOME

# set -A channels 19h 19v 22v 37h 37v 91h 91v

# Spatially interpolate 19h
echo "1
/home/scotts/sic_py/nt_orig/system/0_tbs/NSSSS1DTB01/nssss1d17tb19h2018001" > ${spi_tmpfile}

# /home/scotts/sic_py/nt_orig/bin/SpatialInt_np << args
${spi_exe} << args
19h
${spi_tmpfile}
$SPATIALINT_DIR
args

# Spatially interpolate 19v
echo "1
/home/scotts/sic_py/nt_orig/system/0_tbs/NSSSS1DTB01/nssss1d17tb19v2018001" > ${spi_tmpfile}

/home/scotts/sic_py/nt_orig/bin/SpatialInt_np << args
19v
${spi_tmpfile}
$SPATIALINT_DIR
args

# Spatially interpolate 22v
echo "1
/home/scotts/sic_py/nt_orig/system/0_tbs/NSSSS1DTB01/nssss1d17tb22v2018001" > ${spi_tmpfile}

/home/scotts/sic_py/nt_orig/bin/SpatialInt_np << args
22v
${spi_tmpfile}
$SPATIALINT_DIR
args

# Spatially interpolate 37h
echo "1
/home/scotts/sic_py/nt_orig/system/0_tbs/NSSSS1DTB01/nssss1d17tb37h2018001" > ${spi_tmpfile}

/home/scotts/sic_py/nt_orig/bin/SpatialInt_np << args
37h
${spi_tmpfile}
$SPATIALINT_DIR
args

# Spatially interpolate 37v
echo "1
/home/scotts/sic_py/nt_orig/system/0_tbs/NSSSS1DTB01/nssss1d17tb37v2018001" > ${spi_tmpfile}

/home/scotts/sic_py/nt_orig/bin/SpatialInt_np << args
37v
${spi_tmpfile}
$SPATIALINT_DIR
args

# Clean up
rm ${spi_tmpfile}

### Run the Concentration algorithm

# Create nssss1d17tcon2018001 in conc_dir
cd $conc_dir

${conc_exe} 001 2018 001 2018 TOT_CON ssmif17 n
# ${sst_exe} < ${conc_dir}/nssss1d17tcon2018001 
${sst_exe} << args
${conc_dir}/nssss1d17tcon2018001 
args
mv ${conc_dir}/nssss1d17tcon2018001.spill_sst ${sst_dir}

# Compare with original output
cd $HOME
cmd="cmp -i 300 ${oldfile} ${newfile}"
cmd_output=$($cmd)
echo "  old file: ${oldfile}"
echo "  new file: ${newfile}"
echo "cmd_output: $cmd_output"
if [[ -z $cmd_output ]]; then
  echo "    ... are identical in the data section (after 300b header)"
else
  echo "    ... are DIFFERENT!"
  echo "cmd_output: $cmd_output"
fi
