#/bin/bash

# This script does a batch run on the config file names
# listed after the ls command in the for loop.
# usage: ./batch_run.sh

for file in `ls configs/0604_base_*.conf`
do
	conf=`echo $file | cut -d "/" -f2 | sed 's/.conf//g'`
	echo "Running "$conf"..."
	./run.py $conf
	echo "Done."
	echo
done
