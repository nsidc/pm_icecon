#!/bin/bash

outdir=./a2l1c_out
mkdir -p ${outdir}

python -m pm_icecon.bt.cli a2l1c -d 2022-02-15 -h north -o ${outdir}
