from extract_features  import extract_features as ef
from validation_mc     import validation_mc    as val
from sklearn.svm       import SVC
from sys               import argv
#from matplotlib.pyplot import stem, show

# PARAMETERS:
# ----------
C10 = int( argv[1] )                # --> num_est == Number of Estimators.
rel_gamma = int( argv[2] )          # --> tree_depth == The depth of each of the trees.
seed = 0            # --> seed == The Random Seed of the tree.
num_sets = 10       # --> num_sets == No. of validation sets. The value of K. 10 for CHN-CXR, 11 for MC-CXR.

# START:
# -----
val_correct = 0
val_total   = 0
for val_num in range(num_sets):
    #print 'For validation set', val_num
    Train, Validation = val.fetch_data(val_num)
    
    # TRAINING PART:
    # -------------
    
    (X_train, y_train) = Train
    x_m, y_sd, X_train = ef.normalize(X_train)  # Normalizes the data, extracting mean and SD of training data for normalizing the testing data later.
    
    clf = SVC(kernel = 'rbf', C = C10/10.0, gamma = rel_gamma/10.0, random_state = seed)
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
#print 'Overall validation accuracy:', round(overall,3), '%'
#print # newline.
print str( C10/10.0 ) + ',' + str( rel_gamma/10.0 ) + ',' + str( round(overall, 3) )
