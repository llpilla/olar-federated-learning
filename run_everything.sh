#!/bin/bash

set -x

# Setup for python3 modules
./setup.sh
# Unitary tests to make sure the code is working correctly
python3 unitary_tests.py
# Makespan experiments
./run_all_makespan_experiments.sh
# Timing experiments
./run_all_timing_experiments.sh
# Analysis
./run_analysis_on_new_results.sh
