## Montgomery County X-Ray Set

### What is this?

This is the directory where the validation, training and testing of the **Montgomery County X-Ray set** is performed. It contains the data from features extracted from the MC set. It also contains Python scripts for training, validating and testing the data.  

Lastly, this directory contains some shell scripts for automation, and some CSV files carrying data output. The contents of this directory have been enumerated below:

### Data Subdirectory:

There is a sub-directory in this folder. It is named `mc-cxr`. This folder contains CSV files carrying the data extracted from the MC set. It contains five files in all:

1. The data for all MC set images of class 0 (no TB)  
2. The data for all MC set images of class 1 (abnormal)  
3. The training data, which consists of equal number of both classes, and is shuffled.  
4. The testing data, which consists of equal number of both classes, and is shuffled.  
5. The Python script for shuffling the data in the CSV files.  

### Parameter files:

The files `params_18.py`, `params_105.py`, etc. are parameter files. These contain sets of parameters to be used during training or testing. The file named `params.py` is special. The set of parameters used in this file is used when extracting data for a training fit.  

The file `params_105.py` constains all the features, while the other files contain features of high feature importance as per the Random Forest Classifier.

### Essential Modules:

The following Python scripts define classes and methods that are needed in the other scripts, but should not be called themselves. They are imported in all the other Python scripts.  

1. extract_features.py
2. validation_chn.py
3. validation_mc.py

They define static methods for reading the data from the CSV files, and then provide the static methods for implementing the K-fold cross validation with K = 11.

### Manual scripts:

The following scripts are meant to be run manually. They have options for changing the hyperparameters prior to running.  

**Training and Validation:**  

1. `rf_fixed_kfold.py`: performs k-fold validation on the data set, using the Random Forest classifier.
2. `svm_fixed_kfold_linear.py`: performs k-fold validation on the data set, using the Support Vector Machine classifier of linear kernel.
3. `svm_fixed_kfold_rbf.py`: performs k-fold validation on the data set, using the Support Vector Machine classifier of RBF (Radial Basis Function) kernel.
4. `importance.py`: uses the Random Forest classifier to evaluate the relative importances of the features in the set.

**Testing:**  

1. `rf_fixed_test.py`: tests the Random Forest classifier, training it on the entire training set and then testing it on the available test data.
2. `svm_fixed_test_linear.py`: tests the Support Vector Machine classifier with _Linear_ kernel, training it on the entire training set and then testing it on the available test data.
3. `svm_fixed_test_rbf.py`: tests the Support Vector Machine classifier with _RBF_ kernel, training it on the entire training set and then testing it on the available test data.

### Automatic scripts:

These scripts are not meant to be run manually. They are automatically run by the shell scripts in the directory. They accept `argv` command line arguments, and they must be called from the _bash_ shell. These scripts all have `_auto` appended to their names. Currently, there are two such scripts: `rf_fixed_kfold_auto.py` and `svm_fixed_kfold_auto.py`.

### Shell Scripts:

There are two shell scripts here, one each for the Random Forest and SVM classifier respectively. These shell scripts run the automatic python scripts repeatedly with different hyper-parameters, and save the validation accuracy output into a file named `data.csv`.
