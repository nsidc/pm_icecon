#!/bin/bash

# These are the checks that Circle CI runs when you upload a branch

echo "Run flake8..."
flake8 --exclude nt_tiepoint_generation/
echo " "

echo "Run mypy..."
mypy --config-file=.mypy.ini .
echo " "

echo "Run vulture..."
vulture --exclude tasks/,nt_tiepoint_generation/,pm_icecon/**/_types.py,pm_icecon/nt/_types.py,pm_icecon/config/models/base_model.py,pm_icecon/config/models/__init__.py .
echo " "

echo "Run isort..."
isort --check-only .
echo " "

echo "Run black..."
black --exclude nt_tiepoint_generation/ --check .
echo " "

echo "Run unit tests..."
pytest -s pm_icecon/tests/unit
echo " "
