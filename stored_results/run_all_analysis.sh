#!/bin/bash

set -x

find . -name "Analysis*.py" | while read line; do
    python3 "$line"
done
