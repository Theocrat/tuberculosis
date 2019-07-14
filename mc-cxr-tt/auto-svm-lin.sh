#!/bin/bash

echo C, Validation Accuracy > data.csv

for i in {1..10}
do
    python svm_fixed_kfold_lin_auto.py $i 8 >> data.csv
    echo $i finished
done

