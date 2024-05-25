# hybrid-genetic-optimisation-for-quantum-feature-map-design

## Paper
This project holds the official code for the paper "Hybrid Genetic
Optimisation for Quantum Feature Map Design" which can be found at
[arxiv.org](https://arxiv.org/abs/2302.02980). The diagrams used in
the paper are in `paper_results/diagrams.zip`, and the results used to
produce them are in `paper_results/results.zip`.

## Running the code:

This project makes use of both Python and Julia code to retain library
availability with good performance. As a result, the setup process has
some complexities which are managed by the scripts distributed with
the project. The scripts and code were tested and are working on
Ubuntu 20.04.06 LTS with Python version 3.8.10 and Julia version
1.10.3.

### Steps:
1. Clone / download this repository.

2. Run `install.sh`. On Unix systems, this downloads and installs the
   latest version of Julia to `~/.juliaup` and should be modified if
   you want to avoid this.

3. Run `run.sh` to run the experiments in the paper and save the
   results to the `./results` directory.

4. Run `make_graphs.sh` to generate the result figures used in the
   paper.
