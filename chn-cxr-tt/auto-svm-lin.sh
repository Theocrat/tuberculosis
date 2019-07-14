#!/bin/bash

echo C, gamma, Validation Accuracy > data.csv

for i in {1..10}
do
    for j in {5..10}
    do
        python svm_fixed_kfold_lin_auto.py $i $j >> data.csv
    done
    echo $i finished
done

