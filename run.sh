# activate virtual environment
source activate

# start julia with this project and run the experiments
julia --project="." -e "cd(\"./src\"); include(\"./experiments.jl\"); main()"
