##
## --------------------------------------------------------
##  Nebojsa Djosic  
##  CP8318 Machine Learning - Assignment 2
##  2024-10-25
##  Copyright 2024 Nebojsa Djosic
## --------------------------------------------------------
##  Script will:
##      create and overwrite log file script_Nebojsa_Djosic.log for each run
##      create data and results subdirectories if not exist
##      download 
##      create and save images for all the graphs (will overwrite if exists on each run)
## --------------------------------------------------------
##
import logging
import os
from pathlib import Path
import subprocess
import sys

CPU_CORES = 1  ## <- NOTE WILL BE OVERWRITTEN unless below set to False
AUTO_CPU_CORE = True  ##  Set to False if you want to set the number of CPU cores manually
DATA_DIR = 'data'
RESULTS_DIR = 'results'
PWD = '.'

## configure logging, install dependencies...
if __name__ == '__main__':  
    PWD = Path(__file__).resolve().parent
    DATA_DIR = PWD / 'data'
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR = PWD / 'results'
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[
        logging.FileHandler(RESULTS_DIR / "script_Nebojsa_Djosic.log", mode='w'),
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


msg = """
Assignment 2: regression
Goals: introduction to pandas, sklearn, linear and logistic regression, multi-class classification.
Start early, as you will spend time searching for the proper syntax, especially when using pandas
"""
logging.info(msg)

import pandas as pd
import matplotlib.pyplot as plt

from sklearn import linear_model


msg = """
PART 1: basic linear regression
The goal is to predict the profit of a restaurant, based on the number of habitants where the restaurant 
is located. The chain already has several restaurants in different cities. Your goal is to model 
the relationship between the profit and the populations from the cities where they are located.
Hint: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
"""
logging.info(msg)

# Open the csv file RegressionData.csv in Excel, notepad++ or any other applications to have a 
# rough overview of the data at hand. 
# You will notice that there are several instances (rows), of 2 features (columns). 
# The values to be predicted are reported in the 2nd column.

# Load the data from the file RegressionData.csv in a pandas dataframe. Make sure all the instances 
# are imported properly. Name the first feature 'X' and the second feature 'y' (these are the labels)
# data = pandas._________(_________, header = _________, names=['X', 'y']) # 5 points
logging.debug(f"Loading RegressionData.csv")
data = pd.read_csv(PWD / 'RegressionData.csv', header = None, names=['X', 'y']) # 5 points
# Reshape the data so that it can be processed properly
# X = _________.values.reshape(-1,1) # 5 points
X = data['X'].values.reshape(-1,1) # 5 points
logging.debug(f"X {X[:5]}")
# y = _________ # 5 points
y = data['y'] # 5 points
logging.debug(f"y {y.head()}")

# Plot the data using a scatter plot to visualize the data
#plt._________(_________, _________) # 5 points
plt.scatter(X, y) # 5 points
plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in $10,000s')
plt.title('Profit vs Population')
logging.debug(f"Saving Profit vs Population plot to {RESULTS_DIR / 'lin_reg_prof_v_pop.png'}")
plt.savefig(RESULTS_DIR / 'lin_reg_prof_v_pop.png')
plt.show()

# Linear regression using least squares optimization
# reg = _________.LinearRegression_________() # 5 points
reg = linear_model.LinearRegression() # 5 points
# reg._________(_________, _________) # 5 points
reg.fit(X, y) # 5 points

# Plot the linear fit
fig = plt.figure()
# y_pred = _________._________(_________) # 5 points
y_pred = reg.predict(X) # 5 points
# plt._________(_________,y, c='b') # 5 points
plt.scatter(X, y, c='b') # 5 points
# plt._________(_________, y_pred, 'r') # 5 points
plt.plot(X, y_pred, 'r') # 5 points
fig.canvas.draw()
plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in $10,000s')
plt.title('Profit vs Population')
logging.debug(f"Saving Profit vs Population plot with linear fit to {RESULTS_DIR / 'lin_reg_prof_v_pop_lin_fit.png'}")
plt.savefig(RESULTS_DIR / 'lin_reg_prof_v_pop_lin_fit.png')
plt.show()

# Complete the following print statement (replace the blanks _____ by using a command, do not hard-code the values):
# print(f"The linear relationship between X and y was modeled according to the equation: y = b_0 + X*b_1, \
# where the bias parameter b_0 is equal to ", _________, " and the weight b_1 is equal to ", _________)
msg = f"""
The linear relationship between X and y was modeled according to the equation: 
            y = b_0 + X*b_1
    where 
        - the bias parameter b_0 is equal to {reg.intercept_} and
        - the weight b_1 is equal to {reg.coef_[0]}
"""
logging.info(msg)
# b_0 = -3.89578087831185  b_1 = [1.19303364]
# 8 points

# Predict the profit of a restaurant, if this restaurant is located in a city of 18 habitants 
# print(f"the profit/loss in a city with 18 habitants is ", _________._________(_________))
logging.info(f"\nthe profit/loss in a city with 18 habitants is {reg.predict([[18]])}")
# the profit/loss in a city with 18 habitants is  [17.57882472]
# 8 points
    
msg = """
PART 2: logistic regression 
You are a recruiter and your goal is to predict whether an applicant is likely to get hired or rejected. 
You have gathered data over the years that you intend to use as a training set. 
Your task is to use logistic regression to build a model that predicts whether an applicant is likely to
be hired or not, based on the results of a first round of interview (which consisted of two technical questions).
The training instances consist of the two exam scores of each applicant, as well as the hiring decision.
"""
logging.info(msg)

# Open the csv file in Excel, notepad++ or any other applications to have a rough overview of the data at hand. 

# Load the data from the file 'LogisticRegressionData.csv' in a pandas dataframe. Make sure all the instances 
# are imported properly. Name the first feature 'Score1', the second feature 'Score2', and the class 'y'
# data = _________._________(_________, header = _________, names=['Score1', 'Score2', 'y']) # 2 points
logging.debug(f"Loading LogisticRegressionData.csv")
data = pd.read_csv(PWD / 'LogisticRegressionData.csv', header = None, names=['Score1', 'Score2', 'y']) # 2 points

# Seperate the data features (score1 and Score2) from the class attribute 
# X = _________ # 2 points
X = data[['Score1', 'Score2']] # 2 points
logging.debug(f"X {X.head()}")
# y = _________ # 2 points
y = data['y'] # 2 points
logging.debug(f"y {y.head()}")

# Plot the data using a scatter plot to visualize the data. 
# Represent the instances with different markers of different colors based on the class labels.
m = ['o', 'x']
c = ['hotpink', '#88c999']
fig = plt.figure()
logging.debug(f"Creating Student Scores Scatter Plot...")
for i in range(len(data)):
#     plt.scatter(_________['Score1'][i], _________['Score2'][i], marker=_________[data['y'][i]], color = _________[data['y'][i]]) # 2 points
    plt.scatter(data['Score1'][i], data['Score2'][i], marker=m[data['y'][i]], color = c[data['y'][i]]) # 2 points
fig.canvas.draw()
plt.xlabel('Score1')
plt.ylabel('Score2')
plt.title('Student Scores Scatter Plot')
logging.debug(f"Saving Student Scores Scatter Plot to {RESULTS_DIR / 'log_reg_scores_scatter.png'}")
plt.savefig(RESULTS_DIR / 'log_reg_scores_scatter.png')
plt.show()

# Train a logistic regression classifier to predict the class labels y using the features X
# regS = _________._________() # 2 points
regS = linear_model.LogisticRegression() # 2 points
# regS._________(_________, _________) # 2 points
regS.fit(X, y) # 2 points

# Now, we would like to visualize how well does the trained classifier perform on the training data
# Use the trained classifier on the training data to predict the class labels
# y_pred = _________._________(_________) # 2 points
y_pred = regS.predict(X) # 2 points
# To visualize the classification error on the training instances, we will plot again the data. However, this time,
# the markers and colors selected will be determined using the predicted class labels
m = ['o', 'x']
c = ['red', 'blue'] #this time in red and blue
fig = plt.figure()
logging.debug(f"Creating Preditcted Student Scores Scatter Plot...")
for i in range(len(data)):
    # plt.scatter(_________['Score1'][i], _________['Score2'][i], _________=_________[y_pred[i]], _________ = _________[y_pred[i]]) # 2 points
    plt.scatter(data['Score1'][i], data['Score2'][i], marker=m[y_pred[i]], color = c[y_pred[i]]) # 2 points
fig.canvas.draw()
plt.xlabel('Score1')
plt.ylabel('Score2')
plt.title('Preditcted Student Scores Scatter Plot')
logging.debug(f"Saving Preditcted Student Scores Scatter Plot to {RESULTS_DIR / 'log_reg_scores_scatter_pred.png'}")
plt.savefig(RESULTS_DIR / 'log_reg_scores_scatter_pred.png')
plt.show()
# Notice that some of the training instances are not correctly classified. These are the training errors.

msg = """
PART 3: Multi-class classification using logistic regression 
Not all classification algorithms can support multi-class classification (classification tasks with more than two classes).
Logistic Regression was designed for binary classification.
One approach to alleviate this shortcoming, is to split the dataset into multiple binary classification datasets 
and fit a binary classification model on each. 
Two different examples of this approach are the One-vs-Rest and One-vs-One strategies.
"""
logging.info(msg)

#  One-vs-Rest method (a.k.a. One-vs-All)

# Explain below how the One-vs-Rest method works for multi-class classification # 12 points
msg ="""
One-vs-Rest (OvR) method works by training a single classifier per class.
Each of these classifiers is trained to recognize one single class (positive)
and all other classes (negative).
For instance if we have 3 classes (1, 2, 3), we need to train 3 classifiers:
    - C_1 to recognize class 1 and all other classes
    - C_2 to recognize class 2 and all other classes
    - C_3 to recognize class 3 and all other classes
To predict, we take prediction from each classifier and the class with the highest score
becomes the sole predicted class.
"""
logging.info(msg)

# Explain below how the One-Vs-One method works for multi-class classification # 11 points
msg = """
One-vs-One method works by training a single, binary classifier for each pair of classes.
Each classifier is trained to recognize only two specific classes and ignore all the other ones.
For instance if we have 3 classes (1, 2, 3), we need to train 3 classifiers:
    - C_1(1 vs 2) to recognize class 1 and class 2
    - C_2(1 vs 3) to recognize class 1 and class 3
    - C_3(2 vs 3) to recognize class 2 and class 3
To predict, we take prediction from each classifier and the class with the most votes 
becomes the sole predicted class.
"""
logging.info(msg)

## ---------------------------------------------------------------------------------------##
############## FOR GRADUATE STUDENTS ONLY (the students enrolled in CPS 8318) ##############
## ---------------------------------------------------------------------------------------##

logging.info("\n\nPART 4 FOR GRADUATE STUDENTS ONLY\n\n")
""" 
PART 4 FOR GRADUATE STUDENTS ONLY: Multi-class classification using logistic regression project.
Please note that the grade for parts 1, 2, and 3 counts for 70% of your total grade. The following
work requires you to work on a project of your own and will account for the remaining 30% of your grade.

Choose a multi-Class Classification problem with a dataset (with a reasonable size) 
from one of the following sources (other sources are also possible, e.g., Kaggle):
  •	UCI Machine Learning Repository, https://archive.ics.uci.edu/ml/datasets.php. 
  •	KDD Cup challenges, http://www.kdd.org/kdd-cup.

Download the data, read the description, and use a logistic regression approach to solve a 
classification problem as best as you can. 
Investigate how the One-vs-Rest and One-vs-One methods can help with solving your problem.
Write up a report of approximately 2 pages, double spaced, in which you briefly describe 
the dataset (e.g., the size – number of instances and number of attributes, 
what type of data, source), the problem, the approaches that you tried and the results. 
You can use any appropriate libraries. 

Name your documents appropriately:
report_Firstname_LastName.pdf
script_ Firstname_LastName.py
"""




import matplotlib.pyplot as plt
import numpy as np
import pickle
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo




DIGITS_DATASET_ID = 80  ## Optical Recognition of Handwritten Digits Data Set
DIGITS_DATASET_SHORT_NM = 'digits'
DIGITS_DATASET_NM = 'Optical Recognition of Handwritten Digits'

BEANS_DATASET_ID = 602  ## https://archive.ics.uci.edu/dataset/602/dry+bean+dataset
BEANS_DATASET_SHORT_NM = 'beans'
BEANS_DATASET_NM = 'Optical Recognition of Dry Beans (Grains)'
BEANS_SHORT_DESC ='Images of 13,611 grains of 7 different registered dry beans taken with a high-resolution camera.'
BEANS_LONG_DESC ='''
    t types of dry beans were used in this research, 
    count the features such as form, shape, type, 
    by the market situation. A computer vision system 
    to distinguish seven different registered 
    ry beans with similar features in order to obtain 
    lassification. For the classification model, 
    11 grains of 7 different registered dry beans 
    h a high-resolution camera. Bean images obtained 
    sion system were subjected to segmentation and 
    tion stages, and a total of 16 features; 
    and 4 shape forms, were obtained from the grains.
'''
DATASET_ID = BEANS_DATASET_ID
DATASET_SHORT_NM = BEANS_DATASET_SHORT_NM
DATASET_NM = BEANS_DATASET_NM

FETURES_FILE_NM = 'features.csv'
TARGETS_FILE_NM = 'targets.csv'
RAW_DATASET_FILE_NM = 'raw_dataset.pkl'
FETURES_FILE_NM = 'features.csv'
TARGETS_FILE_NM = 'targets.csv'
RAW_DATASET_FILE_NM = 'raw_dataset.pkl'


##
##  Download and save if not found locally or load from local files
##  Return features and targets as X, y
##
def get_dataset(dataset_id:int=DATASET_ID, 
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
##
def to_float(dict_obj:dict):
    for key, val in dict_obj.items():
        if isinstance(val, dict):
            to_float(val)
        else:
            try:
                dict_obj[key] = float(val)
            except ValueError:
                pass

##
##  
def perf_report(cls_nm:str, y_true, y_pred, data_nm = DATASET_NM, to_latex=True):
    y_true = np.ravel(y_true)
    y_pred = np.ravel(y_pred)
    rpt = classification_report(y_true, y_pred, output_dict=True)
    if(to_latex):
        to_float(rpt)
        rpt_df = pd.DataFrame(rpt).transpose()
        rpt_df = rpt_df.round({
            'precision': 2,
            'recall': 2,
            'f1-score': 2,
            'support': 0  # No decimal places this is a number of samples
        })
        logging.info('Classification Report:{rpt_df}')
        rpt_df.to_latex(RESULTS_DIR / f'{DATASET_ID}_{DATASET_SHORT_NM}_{cls_nm}_cls_rpt.tex', 
                    caption=f'{data_nm} report',
                    index=True)
    else:
        logging.info('Classification Report:')
        logging.info(rpt)

    ##  ------ confusion matrix  ------------
    class_labels = np.unique(y_true)  # Get unique class labels
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
    disp.plot(cmap=plt.cm.Blues, values_format='.2f')
    plt.xticks(rotation=90) ## <- show labels vertically
    # plt.title(f'{data_nm} Confusion Matrix')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f'{DATASET_ID}_{DATASET_SHORT_NM}_{cls_nm}_conf_mtrx.png')
    plt.show()

##
## 
def run_grad_part():
    msg ='''
    
    -----------------------------
     Graduate part of the script
    -----------------------------
    
    '''
    logging.info(msg)

    max_iter = 10000
    random_state = 42

    models = [
        {"name": 'Logistic Regression', "model": LogisticRegression(max_iter=max_iter, random_state=random_state)},
        {"name": 'One Vs One', "model": OneVsOneClassifier(LogisticRegression(max_iter=1000))},
        {"name": 'One Vs Rest', "model": OneVsRestClassifier(LogisticRegression(max_iter=1000))}
    ]

    X, y = get_dataset(DATASET_ID)
    logging.debug(f"Dataset shape: {X.shape}")
    # logging.debug(f"Dataset head: {X.head()}")
    ## Bans:
    ## 13,611 instances of grains (rows)
    ## 16 features (columns): -> 12 dimensions and 4 shape forms
    logging.debug(f"Dataset target shape: {y.shape}")
    # target_names = y.iloc[:, 0].unique()
    # target_names.sort()
    # logging.info(f"Target names: {target_names}")
    ## Beans:
    ## 7 Targets: -> Seker, Barbunya, Bombay, Cali, Dermosan, Horoz and Sira

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    x_train, x_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
    logging.debug(f"Train shape: {x_train.shape}")
    logging.debug(f"Test shape: {x_test.shape}")
    for model in models:
        logging.info(f"Train model: {model['name']}")
        model['model'].fit(x_train, y_train)
        logging.info(f"Predict model: {model['name']}")
        y_pred = model['model'].predict(x_test)
        logging.info(f"Performance report for model: {model['name']}")
        perf_report(model['name'], y_test, y_pred, to_latex=True)

    logging.info('The End of Script')

run_grad_part()