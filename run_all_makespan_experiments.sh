#!/bin/bash

set -x

find . -name "experiment_*.py" | while read line; do
    python3 $line
done
