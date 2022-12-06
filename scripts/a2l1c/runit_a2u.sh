#!/bin/bash

outdir=./bt_out
mkdir -p ${outdir}

python -m pm_icecon.bt.cli amsr2 -d 2022-06-08 -h north -o ${outdir}
