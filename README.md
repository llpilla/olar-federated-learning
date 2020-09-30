# OLAR: OptimaL Assignment of tasks to Resources

OLAR is an algorithm for optimal task assignment in the context of heterogeneous Federated Learning devices.

The Python 3 and Bash scripts can be used to test scheduling algorithms in different scenarios with variations in the number of tasks, resources, kinds of resources (cost functions), lower and upper limits of tasks per resources, etc.

If you wish to reproduce all the results in "*Optimal Task Assignment to Heterogeneous Federated Learning Devices*", just run `./run_everything.sh`.
If you want to reproduce the analysis in said manuscript, check [stored\_results](stored\_results) and run `./run_all_analysis.sh`.

## Dependencies

We use numpy, matplotlib, pandas, seaborn, and scipy.
Please run `./setup` to install them, if needed.

## How to use

If you want to reproduce makespan results only, run `./run_all_makespan_experiments.py`.

If you want to reproduce scheduling time results only, run `./run_all_timing_experiments.py`. Beware that they make that several hours to run, and the computer should be left alone while running these experiments to avoid adding unwanted noise to the results.

If you have reproduced these results and want to analyze then, run `./run_analysis_on_new_results.sh`.

If you want to build new experiments, check the several `experiment` and `timing` files available.
