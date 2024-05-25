# activate virtual environment
source activate

# set number of worker processes that julia should start to run
# experiments
export LOCAL_WORKER_COUNT=$(nproc)

# start the installed julia with this project and run the experiments
~/.juliaup/bin/julia --project="." -e "include(\"./src/experiments.jl\"); main()"
