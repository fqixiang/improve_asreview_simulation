#!/bin/bash

# Test script for all configurations on Abgaz_2023 dataset
# This script tests different combinations of model, oa_status, and abstract_min_length

# Configuration
BENCHMARK="synergy"
DATASET="Abgaz_2023"
N_POS_PRIORS=1
N_NEG_PRIORS=1
N_RANDOM_CYCLES=10

# Arrays of parameters to test
MODELS=("elas_u4" "elas_l2" "elas_h3")
OA_STATUSES=("" "True" "False")  # Empty string means None/all
ABSTRACT_MIN_LENGTHS=("" "100")  # Empty string means None/all

# Counter for tracking progress
total_configs=$((${#MODELS[@]} * ${#OA_STATUSES[@]} * ${#ABSTRACT_MIN_LENGTHS[@]}))
current=0

echo "Testing ${total_configs} configurations for dataset ${DATASET}"
echo "=================================================="

# Loop through all combinations
for model in "${MODELS[@]}"; do
    for oa_status in "${OA_STATUSES[@]}"; do
        for abstract_min_length in "${ABSTRACT_MIN_LENGTHS[@]}"; do
            ((current++))
            
            # Build command arguments
            oa_arg=""
            abs_arg=""
            
            if [ -n "$oa_status" ]; then
                oa_arg="--oa_status $oa_status"
            fi
            
            if [ -n "$abstract_min_length" ]; then
                abs_arg="--abstract_min_length $abstract_min_length"
            fi
            
            # Display configuration
            oa_display=${oa_status:-"all"}
            abs_display=${abstract_min_length:-"all"}
            
            echo ""
            echo "[$current/$total_configs] Testing: model=$model, oa_status=$oa_display, abstract_min_length=$abs_display"
            echo "---"
            
            # Run simulate.py
            echo "Running simulation..."
            uv run ./src/simulate.py \
                --benchmark "$BENCHMARK" \
                --dataset "$DATASET" \
                --model "$model" \
                --n_pos_priors $N_POS_PRIORS \
                --n_neg_priors $N_NEG_PRIORS \
                --n_random_cycles $N_RANDOM_CYCLES \
                $oa_arg \
                $abs_arg
            
            # Check if simulation succeeded
            if [ $? -eq 0 ]; then
                echo "✓ Simulation completed successfully"
                
                # Run tabulate_and_plot.py
                echo "Generating plots and tables..."
                uv run ./src/tabulate_and_plot.py \
                    --benchmark "$BENCHMARK" \
                    --dataset "$DATASET" \
                    --model "$model" \
                    --n_pos_priors $N_POS_PRIORS \
                    --n_neg_priors $N_NEG_PRIORS \
                    $oa_arg \
                    $abs_arg
                
                if [ $? -eq 0 ]; then
                    echo "✓ Plotting completed successfully"
                else
                    echo "✗ Plotting failed"
                fi
            else
                echo "✗ Simulation failed"
            fi
        done
    done
done

echo ""
echo "=================================================="
echo "All configurations tested!"
echo "Check ./results/${BENCHMARK}/${DATASET}/ for outputs"
echo "Check ./results/simulation_summary.csv for summary statistics"
