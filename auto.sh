#!/bin/bash

echo Number of Estimators, Maximum Tree Depth, Overall Validation Accuracy > data.csv

for i in {1..8}
do
    for j in {2..10}
    do
        python rf_fixed_kfold_auto.py $i $j >> data.csv
    done
    echo ' ' >> data.csv
    echo $i Estimators completed.
done
