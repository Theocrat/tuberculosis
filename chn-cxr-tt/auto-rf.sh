#!/bin/bash

echo Number of Estimators, Maximum Tree Depth, Validation Accuracy > data.csv

for i in {2..10}
do
    for j in {1..10}
    do
        python rf_fixed_kfold_auto.py $i $j >> data.csv
    done
    echo $i finished
done

