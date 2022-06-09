#!/bin/bash

set -exuo pipefail

THIS_DIR="$( cd "$(dirname "$0")"; pwd -P )"
export PYTHONPATH=$THIS_DIR/..

python "${THIS_DIR}/../cdr_amsr2/nt/compute_nt_ic.py"
