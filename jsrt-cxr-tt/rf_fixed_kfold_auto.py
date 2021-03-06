from extract_features  import extract_features as ef
from validation_jsrt   import validation_jsrt  as val
from sklearn.ensemble  import RandomForestClassifier
from sys               import argv
from sys               import exit

# PARAMETERS:
# ----------
if len(argv) < 3:
    print 'Incorrect format.'
    print 'Format: python rf_fixed_kfold_auto.py N M'
    print '  - where N is the number of estimators.'
    print '  -   and M is the maximum tree depth.'
    exit(1)
num_est = int( argv[1] )        # --> num_est == Number of Estimators.
tree_depth = int( argv[2] )     # --> tree_depth == The depth of each of the trees.
seed = 0                        # --> seed == The Random Seed of the tree.
#print '\nUsing %s estimators of depth %s.\n' % (str(num_est), str(tree_depth))

# START:
# -----
val_correct = 0
val_total   = 0
for val_num in range(10):
    #print 'For validation set', val_num
    Train, Validation = val.fetch_data(val_num)
    
    # TRAINING PART:
    # -------------
    
    (X_train, y_train) = Train
    x_m, y_sd, X_train = ef.normalize(X_train)  # Normalizes the data, extracting mean and SD of training data for normalizing the testing data later.
    
    clf = RandomForestClassifier(n_estimators = num_est, max_depth = tree_depth, random_state = seed)
    clf.fit(X_train, y_train)
    
    # Finding the Training Accuracy:
    correct = 0
    total   = 0
    
    y_predict_train = clf.predict(X_train)
    for i in range(len(y_predict_train)):
        if list(y_predict_train)[i] == list(y_train)[i]:
            correct += 1
        total += 1
    
    #print '\nTraining accuracy: ' + str(correct) + ' out of ' + str(total) +'.'
    #print 'Training accuracy: ' + str( float(correct * 100) / total ) + '%.'
    
    # ==================================
    
    # TESTING PART:
    # ------------
    
    (X_test, y_test) = Validation
    x_m, y_sd, X_test = ef.normalize(X_test, xm = x_m, ysd = y_sd)  # Normalizes the testing data with the mean and SD of the training set.
       
    # Testing the predictions:
    correct = 0
    total   = 0
    
    y_hat = clf.predict(X_test)
    #print 'Prediction\tActual'
    for i in range(len(y_hat)):
        #print '', list(y_hat)[i], '-' *14, list(y_test)[i]
        if list(y_hat)[i] == list(y_test)[i]:
            correct += 1
        total += 1
    
    val_correct += correct
    val_total   += total
    #print 'Validation Accuracy ' + str(correct) + ' out of ' + str(total) +'.'
    #print 'Validation accuracy: ' + str( float(correct * 100) / total ) + '%.' 
    #print # newline.

overall = float( val_correct ) / val_total
overall = overall * 100
print num_est, ',', tree_depth, ',', round(overall,3)
#print # newline.
