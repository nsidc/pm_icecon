#!/bin/bash

ARGS=$@

THIS_DIR="$( cd "$(dirname "$0")"; pwd -P )"
PYTHONPATH=$THIS_DIR/.. python -m pm_icecon.cli.entrypoint $ARGS
