#/bin/bash

# This script does a batch run on the config file names
# listed after the ls command in the for loop.
# usage: ./batch_run.sh basename

for file in `ls configs/$1_*.conf`
do
	conf=`echo $file | cut -d "/" -f2 | sed 's/.conf//g'`
	echo "Submitting "$conf"..."
	sbatch --exclude=jimgray88 -c 6 --mem=1000 ./run.py $conf
done
