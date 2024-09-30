from sklearn import datasets 
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV

############## FOR EVERYONE ##############
# Please note that the blanks are here to guide you for this first assignment, but the blanks are  
# in no way representative of the number of commands/ parameters or length of what should be inputted.

### PART 1 ###
# Scikit-Learn provides many popular datasets. The breast cancer wisconsin dataset is one of them. 
# Write code that fetches the breast cancer wisconsin dataset. 
# Hint: https://scikit-learn.org/stable/datasets/toy_dataset.html
# Hint: Make sure the data features and associated target class are returned instead of a "Bunch object".
X, y = datasets.load_breast_cancer(return_X_y=True) #(4 points) 

# Check how many instances we have in the dataset, and how many features describe these instances
print("There are",______, "instances described by", ______, "features.") #(4 points)  

# Create a training and test set such that the test set has 40% of the instances from the 
# complete breast cancer wisconsin dataset and that the training set has the remaining 60% of  
# the instances from the complete breast cancer wisconsin dataset, using the holdout method. 
# In addition, ensure that the training and test sets # contain approximately the same 
# percentage of instances of each target class as the complete set.
X_train, X_test, y_train, y_test = ______(______, ______, ______, ______, random_state = 42)  #(4 points) 

# Create a decision tree classifier. Then Train the classifier using the training dataset created earlier.
# To measure the quality of a split, using the entropy criteria.
# Ensure that nodes with less than 6 training instances are not further split
clf = ______(______, ______)  #(4 points) 
clf = ______(______, ______)  #(4 points) 

# Apply the decision tree to classify the data 'testData'.
predC = ______(______)  #(4 points) 

# Compute the accuracy of the classifier on 'testData'
print('The accuracy of the classifier is', ______)  #(2 point) 

# Visualize the tree created. Set the font size the 12 (4 points) 
_ = ______(______,______, ______)  

### PART 2.1 ###
# Visualize the training and test error as a function of the maximum depth of the decision tree
# Initialize 2 empty lists where you will save the training and testing accuracies 
# as we iterate through the different decision tree depth options.
trainAccuracy = ______  #(1 point) 
testAccuracy = ______ #(1 point) 
# Use the range function to create different depths options, ranging from 1 to 15, for the decision trees
depthOptions = ______ #(1 point) 
for depth in ______: #(1 point) 
    # Use a decision tree classifier that still measures the quality of a split using the entropy criteria.
    # Also, ensure that nodes with less than 6 training instances are not further split
    cltree = ______ #(1 point) 
    # Decision tree training
    cltree = ______ #(1 point) 
    # Training error
    y_predTrain = ______ #(1 point) 
    # Testing error
    y_predTest = ______ #(1 point) 
    # Training accuracy
    trainAccuracy.append(______) #(1 point) 
    # Testing accuracy
    testAccuracy.append(______) #(1 point) 

# Plot of training and test accuracies vs the tree depths (use different markers of different colors)
______.______(______,______,______,______,______,______) #(3 points) 
______.______(['Training Accuracy','Test Accuracy']) # add a legend for the training accuracy and test accuracy (1 point) 
______.______('Tree Depth') # name the horizontal axis 'Tree Depth' (1 point) 
______.______('Classifier Accuracy') # name the horizontal axis 'Classifier Accuracy' (1 point) 

# Fill out the following blanks: #(4 points (2 points per blank)) 
""" 
According to the test error, the best model to select is when the maximum depth is equal to ____, approximately. 
But, we should not use select the hyperparameters of our model using the test data, because _____.
"""

### PART 2.2 ###
# Use sklearn's GridSearchCV function to perform an exhaustive search to find the best tree depth and the minimum number of samples to split a node
# Hint: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
# Define the parameters to be optimized: the max depth of the tree and the minimum number of samples to split a node
parameters = {______:______, ______:______} #(6 points)
# We will still grow a decision tree classifier by measuring the quality of a split using the entropy criteria. 
clf = ______(______) #(6 points)
clf.fit(______, ______) #(4 points)
tree_model = clf.______ #(4 points)
print("The maximum depth of the tree sis", __________, 
      'and the minimum number of samples required to split a node is', _______) #(6 points)

# The best model is tree_model. Visualize that decision tree (tree_model). Set the font size the 12 
_ = ______.______(______,filled=True, ______) #(4 points)

# Fill out the following blank: #(2 points)
""" 
This method for tuning the hyperparameters of our model is acceptable, because ________. 
"""

# Explain below what is tenfold Stratified cross-validation?  #(4 points)
"""
______
"""

