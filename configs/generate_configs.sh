#!/bin/bash

# Given a base config file in which we do not modify the parameters,
# this script creates several config files in a loop.
# The created config files inherit the name beginning of the base file,
# and they are located in the same folder.

# Example usage:
# ./generate_configs.sh

base="0606_base.conf"
alg1="baseline_random_user_random_taxi"
alg2="baseline_random_user_nearest_taxi"
alg3="levelling2_random_user_nearest_poorest_taxi_w_waiting_limit"

# generate different taxi to request ratios at N_taxi=200 (corresponding to d = 200 m).

temp=`cat $base | ./add_param.sh -n num_taxis 200`
for lambda in 6 12 25 50
do
	for alg in $alg1 $alg2 $alg3
	do
		output=`echo $base | sed "s/.conf/_fixed_taxis_r_"$lambda"_alg_"$alg".conf/g"`
		echo $temp | ./add_param.sh -n request_rate $lambda | ./add_param.sh -t matching $alg > $output
	done
done

# generate different average distances between taxis (different densities) with fixed R = 0.5

for t in 32 200 800 3200
do
	lambda=`echo "$(("$t"/16))"`
	temp=`cat $base | ./add_param.sh -n num_taxis $t`
	for alg in $alg1 $alg2 $alg3
	do
		output=`echo $base | sed "s/.conf/_fixed_ratio_t_"$t"_alg_"$alg".conf/g"`
		echo $temp | ./add_param.sh -n request_rate $lambda | ./add_param.sh -t matching $alg > $output
	done
done



