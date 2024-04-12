# create python virtual environment and install requirements with pip
python -m venv venv

source venv/bin/activate

pip install -r requirements.txt

julia --project="." -e "ENV[\"PYTHON\"]=\"$(pwd)/venv/bin/python\"; import Pkg; Pkg.instantiate(); Pkg.build(\"PyCall\")"
