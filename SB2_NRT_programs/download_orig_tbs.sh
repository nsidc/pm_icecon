#!/bin/bash

# download_input_tb_files.sh

# Download the NRT TBs used to create the sameple output files provided
# with initial tarball

orig_input_tbs_dir=./orig_input_tbs
mkdir -p ${orig_input_tbs_dir}

scp scotts@nusnow.colorado.edu:/ecs/DP1/PM/NSIDC-0080.001/2018.02.17/*.bin ${orig_input_tbs_dir}
