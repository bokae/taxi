#!/bin/bash

cat benchmark.txt | tail -n +6 | head -n -3 | cut -d" " -f5 | awk 'BEGIN{a=0}{a+=$1}END{print a/NR}'
