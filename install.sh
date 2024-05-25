#!/usr/bin/bash
set -e

# create python virtual environment and install requirements with pip
python -m venv venv

source venv/bin/activate

# install julia (on unix OS's)
curl -fsSL https://install.julialang.org | sh

# install python requirements
pip install -r requirements.txt

# instantiate julia project with requirements, and configure to work
# with the venv python
~/.juliaup/bin/julia --project="." -e "ENV[\"PYTHON\"]=\"$(pwd)/venv/bin/python\"; import Pkg; Pkg.instantiate(); Pkg.build(\"PyCall\")"
