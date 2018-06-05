#!/bin/bash

# Adding a numerical valued or a text valued key value pair
# to a config read from STDIN.

# Can be used for editing config files.
# Writes to standard output.

# Example usage:
# cat simple.conf | ./add_param.sh proba 1


if [ $1 = "-n" ]
then
	cat /dev/stdin | jq --arg key $2 --argjson val $3 '.[$key]=$val'
elif [ $1 = "-t" ]
then
	cat /dev/stdin | jq --arg key $2 --arg val $3 '.[$key]=$val'
else
	echo "Please specify if the parameter has a numeric (-n) or a text (-t) value."
fi

