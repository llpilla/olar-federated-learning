#!/bin/bash

set -x

cp stored_results/Analysis* .

find . -maxdepth 1 -name "Analysis*.py" | while read line; do
    python3 "$line"
done
