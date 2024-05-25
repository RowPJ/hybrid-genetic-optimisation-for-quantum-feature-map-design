# activate virtual environment
source activate

# start the installed julia with this project and run the experiments
~/.juliaup/bin/julia --project="." -e "cd(\"./src\"); include(\"./experiments.jl\"); main()"
