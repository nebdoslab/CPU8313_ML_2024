##
## --------------------------------------------------------
##  Nebojsa Djosic  
##  CP8318 Machine Learning - Assignment 1
##  2024-09-30
##  Copyright 2024 Nebojsa Djosic
## --------------------------------------------------------
##  Script will:
##      create and overwrite log file script_Nebojsa_Djosic.log for each run
##      create data and results subdirectories if not exist
##      download the breast cancer dataset and save it to the data subdirectory (will not overwrite if exists)
##      download the obesity levels dataset and save it to the data subdirectory (will not overwrite if exists)
##      create and save images for all the graphs (will overwrite if exists on each run)
## --------------------------------------------------------
##
import logging
import os
import subprocess
import sys


CPU_CORES = 1  ## <- NOTE WILL BE OVERWRITTEN unless below set to False
AUTO_CPU_CORE = True  ##  Set to False if you want to set the number of CPU cores manually
            

## configure logging, install dependencies...
if __name__ == '__main__':  
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[
        logging.FileHandler("script_Nebojsa_Djosic.log", mode='w'),
        logging.StreamHandler(sys.stdout)
    ])

    ## suppress unnecessary logging...
    matplotlib_logger = logging.getLogger('matplotlib')
    matplotlib_logger.setLevel(logging.WARNING)

    packages = [ ## List of dependencies to install
        "matplotlib",
        "pandas",
        "scikit-learn",
        "ucimlrepo"
    ]
    for package in packages:
        logging.info(f'Installing dependencies: {package}')
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
    # Get the number of CPU cores
    if AUTO_CPU_CORE:
        CPU_CORES = os.cpu_count()
    logging.info(f"Number of CPU cores detected: {CPU_CORES}")



###
import joblib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

from pathlib import Path
from sklearn import datasets 
from sklearn import tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from ucimlrepo import fetch_ucirepo


DATA_DIR = 'data'
RESULTS_DIR = 'results'
if __name__ == '__main__':  ## create local dir to store dataset
    dir = Path(__file__).resolve().parent
    DATA_DIR = dir / 'data'
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR = dir / 'results'
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


##
###################  U N D E R G R A D E  P A R T   ###################
##

################ FOR EVERYONE ##############
### Please note that the blanks are here to guide you for this first assignment, but the blanks are  
### in no way representative of the number of commands/ parameters or length of what should be inputted.

##
#####  -----------------------  PART 1 ----------------
##

### Scikit-Learn provides many popular datasets. The breast cancer wisconsin dataset is one of them. 
### Write code that fetches the breast cancer wisconsin dataset. 
### Hint: https://scikit-learn.org/stable/datasets/toy_dataset.html
### Hint: Make sure the data features and associated target class are returned instead of a "Bunch object".

##
### ---------------------  Dataset Shape ----------------------------------------------------------------------
##
b_cncr_ds_pkl = DATA_DIR / 'breast_cancer_dataset.pkl' # <--- Nebojsa Djosic: local file to avoid downloading the dataset every time

### X, y = datasets.load_breast_cancer(return_X_y=True) #(4 points)  # <--- Nebojsa Djosic: wrapped in a function for caching
def load_dataset(dataset_file_name:str=b_cncr_ds_pkl)->tuple: # <--- Nebojsa Djosic: Add the function to fetch the dataset
    if os.path.exists(b_cncr_ds_pkl):     # <--- Nebojsa Djosic: Check if the local file exists
        X, y = joblib.load(b_cncr_ds_pkl) # <--- Nebojsa Djosic: Load the dataset from the local file
        logging.info(f"Loaded breast cancer dataset from local file {dataset_file_name}.") 
    else: # <--- Nebojsa Djosic: local file does NOT exists
        X, y = datasets.load_breast_cancer(return_X_y=True) # <--- Nebojsa Djosic: Fetch the dataset from the internet
        joblib.dump((X, y), b_cncr_ds_pkl) #  <--- Nebojsa Djosic: Save the dataset locally
        logging.info("Fetched dataset from the internet and saved locally.")
    return X, y

### Check how many instances we have in the dataset, and how many features describe these instances
X, y = load_dataset() # <--- Nebojsa Djosic: Fetch the dataset
num_instances, num_features = X.shape # <--- Nebojsa Djosic Get the number of instances and features
logging.info(f"There are {num_instances} instances described by {num_features} features.") #(4 points)
### Nebojsa Djosic ---> logging.info("There are", 569, "instances described by", 30, "features.")  


##
### ----------------------  Training and Test Split   -----------------------------------------------
##
# ### Create a training and test set such that the test set has 40% of the instances from the 
# ### complete breast cancer wisconsin dataset and that the training set has the remaining 60% of  
# ### the instances from the complete breast cancer wisconsin dataset, using the holdout method. 
# ### In addition, ensure that the training and test sets ### contain approximately the same 
# ### percentage of instances of each target class as the complete set.
# X_train, X_test, y_train, y_test = ______(______, ______, ______, ______, random_state = 42)   

##  Nebojsa Djosic: train_test_split function from scikit-learn implements the holdout method by default.
##  This is the most popular function for splitting the the dataset into training and test sets
##  test_size=0.4 means that the test set will have 40% of the instances from the complete dataset
##  since test_size is 0.4, train_size will be 0.6 (1 - 0.4)
##  random_state=42 ensures reproducibility
##  stratify=y ensures we have approximately the same percentage of each target class as in the complete set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, stratify=y, random_state=42) #(4 points)
logging.info(f"Training set shape: {X_train.shape, y_train.shape}") ## ND -> Training set shape: (341, 30) (341,)
logging.info(f"Test set shape: {X_test.shape, y_test.shape}")  ## ND -> Test set shape: (228, 30) (228,)


##
### ----------------------  Create and Train a Decision Tree Classifier   -------------------------------------
##
# ### Create a decision tree classifier. Then Train the classifier using the training dataset created earlier.
# ### To measure the quality of a split, using the entropy criteria.
# ### Ensure that nodes with less than 6 training instances are not further split
# clf = ______(______, ______)  #(4 points) 
# clf = ______(______, ______)  #(4 points) 
logging.info("Creating and training a decision tree classifier.")
clf = tree.DecisionTreeClassifier(criterion='entropy', min_samples_split=6) # <--- Nebojsa Djosic: Create a decision tree classifier
logging.info("Training the classifier using the training dataset.")
clf = clf.fit(X_train, y_train) # <--- Nebojsa Djosic: Train the classifier using the training dataset

##
### ----------------------  Get Predictions   -------------------------------------
##
# ### Apply the decision tree to classify the data 'testData'.
# predC = ______(______)  #(4 points) 
logging.info("Getting predictions for the test set.")
predC = clf.predict(X_test) # <--- Nebojsa Djosic: Get the predictions for the test set

##
### ----------------------  Accuracy   -------------------------------------
##
# ### Compute the accuracy of the classifier on 'testData'
accuracy = accuracy_score(y_test, predC) # <--- Nebojsa Djosic: Compute the accuracy of the classifier on the test set
logging.info(f'The accuracy of the classifier is {accuracy}')  #(2 point) Accuracy is 0.9254

##
### ----------------------  Visualize the Tree   -------------------------------------
##
# ### Visualize the tree created. Set the font size the 12 (4 points) 
# _ = ______(______,______, ______)  
plt.figure(figsize=(20,10))
_ = tree.plot_tree(clf, filled=True, fontsize=12)
plt.title('Decision Tree Classifier') # <--- Nebojsa Djosic: Add the title to the plot
plt.savefig(RESULTS_DIR / 'ND_asmt_1_part_1_decision_tree.png') # <--- Nebojsa Djosic: Save the tree to a file
logging.info(f"Saved the decision tree image to a file: {RESULTS_DIR / 'ND_asmt_1_part_1_decision_tree.png'}")
plt.show() # <--- Nebojsa Djosic: Visualize the tree created


##### --------------  end of PART 1  --------------


##
###### -----------------------  PART 2.1   ----------------
##

# ### Visualize the training and test error as a function of the maximum depth of the decision tree

##
### ---------------------- Collect Accuracy Scores  -------------------------------------
##
# ### Initialize 2 empty lists where you will save the training and testing accuracies 
# ### as we iterate through the different decision tree depth options.
# trainAccuracy = ______  #(1 point) 
trainAccuracy = []
# testAccuracy = ______ #(1 point) 
testAccuracy = []
# ### Use the range function to create different depths options, ranging from 1 to 15, for the decision trees
# depthOptions = ______ #(1 point) 
depthOptions = range(1, 16)
# for depth in ______: #(1 point) 
logging.info(f"Using Decision Tree Classifier with different depths ranging from 1 to 15.")
for depth in depthOptions:
#     ### Use a decision tree classifier that still measures the quality of a split using the entropy criteria.
#     ### Also, ensure that nodes with less than 6 training instances are not further split
#     cltree = ______ #(1 point) 
    cltree = tree.DecisionTreeClassifier(criterion='entropy', min_samples_split=6, max_depth=depth)
#     ### Decision tree training
#     cltree = ______ #(1 point) 
    cltree.fit(X_train, y_train)
#     ### Training error
#     y_predTrain = ______ #(1 point) 
    y_predTrain = cltree.predict(X_train)
#     ### Testing error
#     y_predTest = ______ #(1 point) 
    y_predTest = cltree.predict(X_test)
#     ### Training accuracy
#     trainAccuracy.append(______) #(1 point) 
    trainAccuracy.append(accuracy_score(y_train, y_predTrain))
#     ### Testing accuracy
#     testAccuracy.append(______) #(1 point) 
    testAccuracy.append(accuracy_score(y_test, y_predTest))

#
### ----------------------  Accuracy Scores vs Depth  -------------------------------------
##
# ### Plot of training and test accuracies vs the tree depths (use different markers of different colors)
# ______.______(______,______,______,______,______,______) #(3 points) 
# ______.______(['Training Accuracy','Test Accuracy']) ### add a legend for the training accuracy and test accuracy (1 point) 
# ______.______('Tree Depth') ### name the horizontal axis 'Tree Depth' (1 point) 
# ______.______('Classifier Accuracy') ### name the horizontal axis 'Classifier Accuracy' (1 point) 
plt.plot(depthOptions, trainAccuracy, marker='o', color='b', label='Training Accuracy')
plt.plot(depthOptions, testAccuracy, marker='x', color='r', label='Test Accuracy')
plt.legend(['Training Accuracy','Test Accuracy'])
plt.xlabel('Tree Depth')
plt.ylabel('Classifier Accuracy')
plt.title('Tree Depth vs Classifier Accuracy')
plt.savefig(RESULTS_DIR / 'ND_asmt_1_part_2-1_tree_depth_vs_accuracy.png') # <--- Nebojsa Djosic: Save the plot to a file
logging.info(f"Saved the accuracy plot to a file: {RESULTS_DIR / 'ND_asmt_1_part_2-1_tree_depth_vs_accuracy.png'}")
plt.show() # <--- Nebojsa Djosic: Visualize the plot

train_errors = [1 - acc for acc in trainAccuracy]
test_errors = [1 - acc for acc in testAccuracy]
plt.plot(depthOptions, train_errors, marker='o', color='b', label='Training Error')
plt.plot(depthOptions, test_errors, marker='x', color='r', label='Test Error')
plt.xlabel('Tree Depth')
plt.ylabel('Error Rate')
plt.legend()
plt.title('Tree Depth vs Error Rate')
plt.savefig(RESULTS_DIR / 'ND_asmt_1_part_2-1_tree_depth_vs_error.png')
logging.info(f"Saved the error rate plot to a file: {RESULTS_DIR / 'ND_asmt_1_part_2-1_tree_depth_vs_error.png'}")
plt.show()

# ### Fill out the following blanks: #(4 points (2 points per blank)) 
##  NOTE:  Nebojsa Djosic: the "test error" rate is 1 - test accuracy thus would be inverted graph
##         however, the resulting maximum depth would still be the same
msg =""" 
According to the test error, the best model to select is when the maximum depth is equal to 3, approximately. 
But, we should not use select the hyperparameters of our model using the test data, because it could lead to overfitting.
"""
logging.info(msg)

##
### ----------------------  End of PART 2.1   -------------------------------------
##


##
### ----------------------  PART 2.2   -------------------------------------
##
# ##### PART 2.2 ###


X, y = load_dataset()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, stratify=y, random_state=42)

# ### Use sklearn's GridSearchCV function to perform an exhaustive search to find the best tree depth and the minimum number of samples to split a node
# ### Hint: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
# ### Define the parameters to be optimized: the max depth of the tree and the minimum number of samples to split a node
# parameters = {______:______, ______:______} #(6 points)
parameters = {'max_depth': range(1, 16), 'min_samples_split': range(2, 11)} 
# ### We will still grow a decision tree classifier by measuring the quality of a split using the entropy criteria. 
# clf = ______(______) #(6 points)
logging.info("Using GridSearchCV to find the best hyperparameters for the decision tree classifier. Max depth = range(1, 16) and min samples split = range(2, 11).") 
clf = GridSearchCV(tree.DecisionTreeClassifier(criterion='entropy'), parameters, cv=10)
# clf.fit(______, ______) #(4 points)
logging.info("Fitting the classifier using the training dataset.")
clf.fit(X_train, y_train)
# tree_model = clf.______ #(4 points)
tree_model = clf.best_estimator_
# logging.info("The maximum depth of the tree sis", __________, 
#       'and the minimum number of samples required to split a node is', _______) #(6 points)
max_dpt = tree_model.get_params()['max_depth']
min_splt = tree_model.get_params()['min_samples_split']
logging.info(f"The maximum depth of the tree is {max_dpt}, and the minimum number of samples required to split a node is {min_splt}")

# ### The best model is tree_model. Visualize that decision tree (tree_model). Set the font size the 12 
# _ = ______.______(______,filled=True, ______) #(4 points)
plt.figure(figsize=(20,10))
_ = tree.plot_tree(tree_model, filled=True, fontsize=12)
plt.title('Decision Tree Classifier with GridSearchCV') 
plt.savefig(RESULTS_DIR / 'ND_asmt_1_part_2-2_decision_tree_gridsearch.png')
logging.info(f"Saved the decision tree image to a file: {RESULTS_DIR / 'ND_asmt_1_part_2-2_decision_tree_gridsearch.png'}")
plt.show()

# ### Fill out the following blank: #(2 points)
msg = """ 
This method for tuning the hyperparameters of our model is acceptable, because:
    - It is an exhaustive search of the hyperparameters, all possible combinations are tried
    - It uses cross-validation to avoid overfitting
    - It is reproducible due to the random_state parameter so we can compare
    - It can handle the class imbalance handled by stratified cross-validation
    - Brest cancer dataset we're using is small so the computational cost is not too high
"""
logging.info(msg)
# ### Explain below what is tenfold Stratified cross-validation?  #(4 points)
msg = """
Tenfold stratified cross-validation is used to evaluate the performance of a model.
The dataset is divided into 10 equal parts, and then the model is trained and tested 10 times.
Each time, only one of the 10 is used for testing, and all the rest for training.
All the performance metrics like accuracy are averaged over all 10 itterations (folds).
Stratified means that each dataset part has the same class distribution as the whole dataset.
In the scikit-learn library this is implemented in the StratifiedKFold(n_splits=10) and then
cross_val_score(...) provides scors and we can print metrics.
"""
logging.info(msg)

##
### ----------------------  End of PART 2.2  (total 86 points, should be 85)  -------------------------------------
##


##########################     -----------------     ##############################


##
### ----------------------  PART 3  (Total 15 points)  -------------------------------------
##
logging.info("Starting PART 3, for details see the report.")
# X1 and X2 are features, Y is the target variable
X = np.array([
    [0, 0],
    [0, 0],
    [1, 0],
    [1, 0],
    [0, 1],
    [0, 1],
    [1, 1],
    [1, 2],
    [0, 2],
    [1, 2]
])
Y = np.array([True, True, True, True, False, False, True, True, False, False])

# Create the decision tree classifier
clf = DecisionTreeClassifier(criterion='entropy')  # You can also use 'gini' as the criterion
clf.fit(X, Y)

##
##  -------------  Question 1 (3 points) -------------
##

## Calculations for probability and entropy
p_true = np.sum(Y) / len(Y)
p_false = 1 - p_true
entropy = - (p_true * np.log2(p_true) + p_false * np.log2(p_false))
logging.info(f"Entropy of training examples = {entropy:.4f}")

msg ='''
Question 1: What is the entropy of this collection of training examples with respect to the target class? (3 points)

Answer:
Entropy of training examples = 0.9710
Steps:
1. Count the number of instances for each class:
    Number of True instances: 6
    Number of False instances: 4
    Total number of instances: 10

2.Calculate the probabilities:
    Probability of True p_(T): 6 / 10 = 0.6
    Probability of False p_(F): 4 / 10 = 0.4 

3. Apply the entropy formula: 
     = - ( 0.6 * log_2(0.6) + 0.4 * log_2(0.4) )
     = - ( 0.6 * -0.7369655941662062 + 0.4 * -1.3219280948873624) 
     = - ( -0.44217935649972373 - 0.5287712379549449 )
     = 0.9709505944546686

The entropy of the collection with respect to the target class is approximately 0.971.
'''
logging.info(msg)

##
##  -------------  Question 2 (3 points) -------------
##
splits = [  # Possible splits for the first split: (feature index, threshold)
    (0, 0),  # X1 = 0 vs. X1 = 1
    (1, 0),  # X2 = 0 vs. X2 != 0
    (1, 1),  # X2 = 1 vs. X2 != 1
    (1, 2)   # X2 = 2 vs. X2 != 2
]

msg = '''
Question 2: What are the different options for the first split when constructing your decision tree?

Answer (see report for more details):
    Features and Values:
        Feature X1: Binary attribute with values {0, 1}
        Feature X2: Ordinal categorical attribute with values {0, 1, 2}
    Possible Splits:
        Splits on X1 are clear since it is a binary attribute 0 or 1:
            X1 = 0 or X1 = 1
        Splits on X2 have more options since the values are 0, 1, and 2:
            X2 = 0 or X2 != 0: either the value is 0 or not in which case it is 1 or 2
            X2 = 1 vs. X2 != 1: either the value is 1 or not in which case it is 0 or 2
            X2 = 2 vs. X2 != 2: either the value is 2 or not in which case it is 0 or 1
'''
logging.info(msg)

##
##  -------------  Question 3 (3 points) -------------
##

def entropy(y): ## See report for more details
    p = np.bincount(y) / len(y)
    return -np.sum([p_i * np.log2(p_i) for p_i in p if p_i > 0])

##
##
def information_gain(X, y, feature_i, threshold):  ## See report for more details
    parent_ent = entropy(y)
    true_i = X[:, feature_i] == threshold    # True instances at the split
    false_i = X[:, feature_i] != threshold   # False instances at the split
    true_ent = entropy(y[true_i])            # Entropy of the true instances
    false_ent = entropy(y[false_i])          # Entropy of the false instances
    n = len(y)                               # Total number of instances
    n_true = np.sum(true_i)                  # Number of true instances
    n_false = np.sum(false_i)                # Number of false instances
    
    child_ent = (n_true / n) * true_ent + (n_false / n) * false_ent
    
    return parent_ent - child_ent

logging.info(f"Calculating information gain for each potential first split options: {splits}")
##
for feature_index, threshold in splits: ## Calculate info gain for each potential first split
    gain = information_gain(X, Y, feature_index, threshold)
    logging.info(f"Information gain for split on feature {feature_index} with threshold {threshold}: {gain:.3f}")

msg = '''
Question 3: For each potential first split option, compute the information gain. Only provide the results,
there is no need to provide your calculations

Answer:
    Information gain for split on X1 = 0, threshold 0 (vs. X1 = 1): 0.125
    Information gain for split on X2 = 0, threshold 0 (X2 != 0): 0.420
    Information gain for split on X2 = 1, threshold 1 (X2 != 1): 0.091
    Information gain for split on X2 = 2, threshold 2 (X2 != 2): 0.091
'''
logging.info(msg)

##
##  -------------  Question 4 (6 points) -------------
##
msg ='''
Question 4: Build the complete decision tree based on the given specifications and training set. The
representation of the tree should adhere to the style used in the lecture notes of this course. 

Answer:
    See the decision tree visualization in the report.
'''
logging.info(msg)

## 
# plt.figure(figsize=(10, 6))
# plot_tree(clf, filled=True, feature_names=['x1', 'x2'], class_names=['False', 'True'], fontsize=12)
# plt.title('Decision Tree Classifier')
# plt.savefig('ND_asmt_1_part_3_decision_tree.png')
# plt.show()

##
#### ----------------------  End of PART 3  (Running Total: 82 + 15 = 97 points)  -------------------------------------
##

logging.info("End of the undergraduate parts (Parts 1-3).")


##
###################  GRADUATE PART  ###################
##

logging.info("Starting the graduate part: Part 4.")

OBISITY_LEVELS_DATASET_ID = 544 # ID of the obesity levels dataset in the UCI repository
FETURES_FILE_NM = 'features.csv'
TARGETS_FILE_NM = 'targets.csv'
RAW_DATASET_FILE_NM = 'raw_dataset.pkl'


##
##  Download and save if not found locally or load from local files
##  Return features and targets as X, y
##
def get_dataset(dataset_id:int=OBISITY_LEVELS_DATASET_ID, 
                raw_dataset_file:str=RAW_DATASET_FILE_NM,
                features_file:str=FETURES_FILE_NM, 
                targets_file:str=TARGETS_FILE_NM, 
                data_dir:str=DATA_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    
    raw_dataset_file = data_dir / f'{dataset_id}_{raw_dataset_file}'
    features_file = data_dir / f'{dataset_id}_{features_file}'
    targets_file = data_dir / f'{dataset_id}_{targets_file}'
    if os.path.exists(features_file) and os.path.exists(targets_file):
        # with open(dataset_file, 'rb') as file: ## TODO: remove go for X, y directly
        #     dataset_raw = pickle.load(file)
        logging.info(f'Loading dataset from local files: features {features_file} and targets {targets_file}')
        X = pd.read_csv(features_file)
        y = pd.read_csv(targets_file)
    else:
        logging.info(f'Downloading dataset {dataset_id} from UCI repository')
        dataset_raw = fetch_ucirepo(id=dataset_id)
        logging.info(f'loaded metadata: {dataset_raw.metadata}')
        logging.info(f'loaded variables: {dataset_raw.variables}')
        logging.info(f'saving raw dataset to a local file: {raw_dataset_file}')
        with open(raw_dataset_file, 'wb') as file:  ## TODO: make it conditional or remove
            pickle.dump(dataset_raw, file)
        X = dataset_raw.data.features
        y = dataset_raw.data.targets
        logging.info(f'saving features and targets to local files: features {features_file} and targets {targets_file}')
        X.to_csv(features_file, index=False)
        y.to_csv(targets_file, index=False)
    return X, y

##
##  Save dataframe to a latex file
def df_to_latex_file(df:pd.DataFrame, file_name:str, caption:str=None) -> None:
    ## TODO add latex label for referencing
    ## TODO add some formatting options...
    with open(file_name, 'w') as file:
        file.write(df.to_latex(index=False, caption=caption))
##
### split wide dataframe ... latex
def df_to_latex_file_wide(df: pd.DataFrame, file_name: str, caption: str = None) -> None:
    ## TODO add latex label for referencing
    mid_point = len(df.columns) // 2 ## TODO: make dynamic...
    df1 = df.iloc[:, :mid_point]
    df2 = df.iloc[:, mid_point:]
    with open(file_name, 'w') as file:
        file.write("\\begin{table}[ht]\n")
        file.write("\\centering\n")
        if caption:
            file.write(f"\\caption{{{caption} (Part 1)}}\n")
        file.write(df1.to_latex(index=False))
        file.write("\\end{table}\n")
        file.write("\\begin{table}[ht]\n")
        file.write("\\centering\n")
        if caption:
            file.write(f"\\caption{{{caption} (Part 2)}}\n")
        file.write(df2.to_latex(index=False))
        file.write("\\end{table}\n")


###
def create_dataset_latex_table(df: pd.DataFrame, caption: str = None) -> str:
    sample_x_file_nm = RESULTS_DIR / f'{OBISITY_LEVELS_DATASET_ID}_obesity_features.tex'
    df_to_latex_file(X.head(), sample_x_file_nm, 'Obesity levels dataset sample features')
    sample_x_file_nm = RESULTS_DIR / f'{OBISITY_LEVELS_DATASET_ID}_obesity_features_wide.tex'
    df_to_latex_file_wide(X.head(), sample_x_file_nm, 'Obesity levels dataset sample features')


###
def get_train_test(dataset_id:int=OBISITY_LEVELS_DATASET_ID) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list]:

    ## TODO: extract to a function, externalize to a config file, so specify ID, etc. the rest is dynamic
    train_x_file = DATA_DIR / f'{OBISITY_LEVELS_DATASET_ID}_train_x.csv'
    train_y_file = DATA_DIR / f'{OBISITY_LEVELS_DATASET_ID}_train_y.csv'
    test_x_file = DATA_DIR / f'{OBISITY_LEVELS_DATASET_ID}_test_x.csv'
    test_y_file = DATA_DIR / f'{OBISITY_LEVELS_DATASET_ID}_test_y.csv'
    y_labels_file = DATA_DIR / f'{OBISITY_LEVELS_DATASET_ID}_y_labels.pkl'

    if( os.path.exists(train_x_file)    # NOTE: even if one file is missing, we will re-create all
       and os.path.exists(train_y_file) # makes it easier when dataset is small
       and os.path.exists(test_x_file) 
       and os.path.exists(test_y_file)
       and os.path.exists(y_labels_file)
    ):
        logging.info('Train and test datasets already saved to local files')
        # load train and test datasets
        x_train = pd.read_csv(train_x_file)
        x_test = pd.read_csv(test_x_file)
        y_train = pd.read_csv(train_y_file)
        y_test = pd.read_csv(test_y_file)
        y_labels = pickle.load(open(y_labels_file, 'rb'))
    else:  # create train and test datasets
        logging.info('getting dataset...')
        X, y = get_dataset()
        logging.info(f'X shape (instances, features): {X.shape}, y shape: {y.shape}')
        logging.info(f'Sample X values (instances): {X.head()}')
        target_names = y.iloc[:, 0].unique()
        logging.info(f'Distinct y values (classes): {target_names}')
        ## TODO: make configuratble....
        if(False):create_dataset_latex_table(X)
        if(False):df_to_latex_file(pd.DataFrame(target_names, columns=['Targets']), RESULTS_DIR / f'{OBISITY_LEVELS_DATASET_ID}_obesity_targets.tex', 'Obesity levels dataset targets')
        ## TODO: extract encoding to a function and make config driven
        logging.info('Encoding categorical features')
        X_encoded = pd.get_dummies(X)
        X_encoded.head().to_csv(DATA_DIR / f'{OBISITY_LEVELS_DATASET_ID}_obesity_features_encoded.csv', index=False)
        le = LabelEncoder()
        y_encoded = le.fit_transform(y.iloc[:, 0])
        y_encoded = pd.DataFrame(y_encoded, columns=['Target'])
        y_encoded.to_csv(DATA_DIR / f'{OBISITY_LEVELS_DATASET_ID}_obesity_targets_encoded.csv', index=False)
        y_labels = le.classes_
        pickle.dump(y_labels, open(y_labels_file, 'wb'))
        logging.info('Splitting dataset into train and test datasets')
        x_train, x_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=0.2, random_state=42, stratify=y)
        logging.info(f'Train shape: {x_train.shape}, Test shape: {x_test.shape}')
        # save train and test datasets
        x_train.to_csv(train_x_file, index=False)
        x_test.to_csv(test_x_file, index=False)
        y_train.to_csv(train_y_file, index=False)
        y_test.to_csv(test_y_file, index=False)
        logging.info('Train and test datasets saved to local files')

    logging.info(f'Train shape: {x_train.shape}, Test shape: {x_test.shape}')
    return x_train, x_test, y_train, y_test, y_labels


##
## Get the best classifier using GridSearchCV
def get_best_clf(x_train, y_train, cpu_cores=CPU_CORES) -> DecisionTreeClassifier:
    param_grid = {
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': [None, 'sqrt', 'log2'],
        'criterion': ['gini', 'entropy'],
        'splitter': ['best', 'random']
    }
    logging.info(f'Parameters grid for GridSearchCV: {param_grid}')
    msg = f'''
        Constructor: GridSearchCV(
                estimator=clf, 
                param_grid=param_grid, 
                cv=10,    ## 10-fold cross validation
                n_jobs={cpu_cores}, ##  NOTE: set to the number of cores
                verbose=2)
    '''
    logging.info(msg)
    logging.info('Looking for the best hyperparameters for the decision tree classifier with GridSearchCV .... ')
    clf = DecisionTreeClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, 
                               cv=10,    ## 10-fold cross validation
                               n_jobs=cpu_cores, ##  NOTE: set to the number of cores
                               verbose=2)
    grid_search.fit(x_train, y_train)
    best_params = grid_search.best_params_
    logging.info(f'Best parameters found: {best_params}')
    best_clf = grid_search.best_estimator_
    return best_clf



##
def run_experiment(clf, x_train, x_test, y_train, y_test, y_lables, to_latex=False):
    ## TODO: create functions for all these below....
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    ## ----- performance metrics ------------
    rpt = classification_report(y_test, y_pred, output_dict=to_latex, target_names=y_lables)
    if(to_latex):
        rpt_df = pd.DataFrame(rpt).transpose()
        rpt_df.to_latex(RESULTS_DIR / f'{OBISITY_LEVELS_DATASET_ID}_obesity_classification_report_manual.tex', 
                        caption='Obesity levels classification report',
                        index=True)
    else:
        logging.info('Classification Report:')
        logging.info(rpt)

    ##  ------ confusion matrix  ------------
    cm = confusion_matrix(y_test, y_pred, normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=y_lables)
    disp.plot(cmap=plt.cm.Blues, values_format='.2f')
    plt.xticks(rotation=90) ## <- Obesity levels are long, show them vertically
    plt.title('Obesity Levels Confusion Matrix')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f'{OBISITY_LEVELS_DATASET_ID}_obesity_confusion_matrix.png')
    plt.show()

    ##  ------ feature importances  ------------
    feature_importances = clf.feature_importances_
    features = x_train.columns
    importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    logging.info('Feature Importances:')
    logging.info(importance_df)
    
    plt.figure(figsize=(10, 6))
    plt.barh(importance_df['Feature'], importance_df['Importance'])
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Feature Importances')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f'{OBISITY_LEVELS_DATASET_ID}_obesity_feature_importances.png')
    plt.show()

    ##  ---------  Plot decision tree    ------------
    plt.figure(figsize=(20,10))
    plot_tree(clf, filled=True, feature_names=x_train.columns, class_names=y_lables)
    plt.savefig(RESULTS_DIR / f'{OBISITY_LEVELS_DATASET_ID}_obesity_decision_tree.png') 
    plt.show()




##
##
if __name__ == '__main__':
    ## TODO get config file name from input args ...
    logging.info('Using manually set parameters for the decision tree classifier:')
    msg = '''
        random_state=42,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=3,
        max_features='sqrt',
        criterion='entropy',
        splitter='best'
    '''
    logging.info(msg)
    clf = DecisionTreeClassifier(
        random_state=42,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=3,
        max_features='sqrt',
        criterion='entropy',
        splitter='best'
    )
    x_train, x_test, y_train, y_test, y_lables = get_train_test()
    run_experiment(clf, x_train, x_test, y_train, y_test, y_lables)
    ## TODO this is ugly...
    x_train, x_test, y_train, y_test, y_lables = get_train_test()
    logging.info('Using GridSearchCV to find the best hyperparameters for the decision tree classifier')

    ## NOTE: override CPU_CORES as needed:
    run_experiment(get_best_clf(x_train, y_train, CPU_CORES), x_train, x_test, y_train, y_test, y_lables)

    logging.info('End of the graduate part (Part 4).')