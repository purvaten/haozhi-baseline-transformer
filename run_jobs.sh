#!/bin/bash

# # TRAIN
# for i in "0 20000" "20000 40000" "40000 60000" "60000 80000" "80000 100000" "100000 120000" "120000 140000" "140000 160000" "160000 180000" "180000 200000"
# do
#     set -- $i
#     sbatch launch_jobs.sh ${1} ${2}
# done


# TEST
for i in "0 10000" "10000 20000" "20000 30000" "30000 40000" "40000 50000"
do
    set -- $i
    sbatch launch_jobs.sh ${1} ${2}
done