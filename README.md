# ASReview Simulation

This repository contains scripts and resources for running simulations and analyses using the ASReview library for the IMPROVE project. Several directories are available, each serving a specific purpose.

## Directory Structure
- **data/**: Includes datasets in `.xlsx` format for various disease topics such as chronic rhinosinusitis and macular degeneration.
- **logs/**: Stores error and output logs generated during batch runs.
- **results/**: Organized results of simulations for different datasets (i.e., disease topics) and prior configurations (i.e., number of positive and negative priors).
- **src/**: Source code for simulation, tabulation and plotting scripts.

## Key Files
- **run_batch.sh**: A script to execute batch simulations, tabulations and plotting. Ensure it has executable permissions (`chmod +x run_batch.sh`) before running.
- **pyproject.toml**: Configuration file for managing dependencies and project settings.
- **README.md**: This file, providing an overview of the project.

## How to Use
1. **Set Up the Environment**:
   - Install environment and dependencies. We recommend using `uv` with `uv sync`. 

2. **Run Simulations**:
   - Use the `run_batch.sh` script to execute batch simulations.
   - Check the `logs/` directory for output and error logs.

3. **Analyze Results**:
   - Inspect the results stored in the `results/` directory.
   - Use scripts in the `src/` directory for plotting and further analysis.

## Dependencies
Ensure the following dependencies are installed:
- Python 3.10 or higher
- Required Python packages

## Contact
For questions or issues, please contact the project maintainer.