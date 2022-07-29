#!/bin/bash

work_dir=$3
mkdir -p $work_dir
rm -f ${work_dir}/*
cd $work_dir
cp ../polymerize_og.py ../5mer_for_polymerization.xyz .

touch polymer.inp
echo "tolerance 2.0
filetype xyz
output system.xyz" >> polymer.inp

nchains=$1
chain_len=$2

for ((i=1;i<=$nchains;i++))
do
	./polymerize_og.py $chain_len mpim_${i}
	echo "structure mpim_${i}.xyz
	number 1
	inside cube 0. 0. 0. 350.
end structure" >> polymer.inp
done

packmol < polymer.inp
rm polymer.inp mpim_*.xyz polymerize_og.py 5mer_for_polymerization.xyz
