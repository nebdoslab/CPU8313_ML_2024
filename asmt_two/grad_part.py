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
        "ucimlrepo",
        "Jinja2"
    ]
    for package in packages:
        logging.info(f'Installing dependencies: {package}')
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
    # Get the number of CPU cores
    if AUTO_CPU_CORE:
        CPU_CORES = os.cpu_count()
    logging.info(f"Number of CPU cores detected: {CPU_CORES}")



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
if __name__ == '__main__':

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