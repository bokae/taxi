#!/bin/bash

cat benchmark.txt | grep Simulation | cut -d" " -f5 | awk 'BEGIN{a=0}{a+=$1}END{print a/NR}'
