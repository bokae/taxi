#/bin/bash

# This script does a batch run on the config file names
# listed after the ls command in the for loop.
# usage: ./batch_run.sh basename

#for file in `ls configs/$1_*.conf`

# more fancy: only runs left out pieces
#for conf in `cat slurm-* | grep run | cut -d " " -f5 | sed 's/\.//g' | sort -n | comm -3 - configs/0831_all.list`
#for file in `ls configs/1114_base*_true*`

#for file in `ls configs/1114_base*_stay*home*_false*`
#do
#	conf=`echo $file | cut -d "/" -f2 | sed 's/.conf//g'`
#	echo "Submitting "$conf"..."
#	sbatch -c 1 --mem=1000 ./run.py $conf
#done

#for file in `ls configs/1114_base*_stay*base*_false*`
#do
#	conf=`echo $file | cut -d "/" -f2 | sed 's/.conf//g'`
#	echo "Submitting "$conf"..."
#	sbatch -c 1 --mem=1000 ./run.py $conf
#done

for alg in "random_limited" "nearest" "poorest" "random_unlimited"
do
	for file in `ls configs/*$alg*`
	do
		conf=`echo $file | cut -d "/" -f2 | sed 's/.conf//g'`
		echo "Submitting "$conf"..."
		sbatch -c 1 --mem=1000 ./run.py $conf
		sleep 1
	done
done
