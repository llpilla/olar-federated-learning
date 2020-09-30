#!/bin/bash

set -x

find . -name "timing_*.py" | while read line; do
    python3 $line
done
