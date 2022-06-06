#!/bin/csh

# original ksh:
#set -A files apply_sst_n apply_sst_s SpatialInt_np SpatialInt_sp SpatialInt_np85  SpatialInt_sp85
#
#for file in ${files[*]}
#do
#  cc -o $file $file.c
#  mv $file ../bin
#done


# set files=(apply_sst_n apply_sst_s SpatialInt_np SpatialInt_sp SpatialInt_np85  SpatialInt_sp85)
set files=(apply_sst_n apply_sst_s SpatialInt_np SpatialInt_sp)

echo $files[1]
echo "Total number of files $#files"

set i = 1
while ($i <= $#files)
 echo "Compiling $files[$i].c"
 # cc -o $files[$i] $files[$i].c
 gcc -o $files[$i] $files[$i].c
 mv $files[$i] ../bin
 @ i = $i + 1
end

make -f makefile_lookup
make -f makelibice5_ts
make -f makeseaice5con
