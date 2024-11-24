##
## --------------------------------------------------------
##  Nebojsa Djosic  
##  CP8318 Machine Learning - Assignment 4
##  2024-11-27 (due date)
##  Copyright 2024 Nebojsa Djosic
##
##  created with python 3.12.3
##  to run the script first create and activate python environment
##  $  cd <dir where the script is>
##  $  python -m venv venv
##  $  source venv/bin/activate
## --------------------------------------------------------
##

import logging
import os
import subprocess
import sys
from pathlib import Path

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

    ## This is typically done using requirements.txt 
    ## made available through GitHub repository,
    ## but given the submission reuqirements for this assignment, 
    ## we will have to install dependencies here:
    packages = [ ## List of dependencies to install
        "gower",
        "matplotlib",
        "pandas",
        "scikit-learn",
        "ucimlrepo"
    ]
    for package in packages:
        logging.info(f'Installing dependencies: {package}')
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

msg = """
Assignment 4: Clustering with Preprocessing
Choose a dataset (as opposed to the example ones we used in class), requiring pre-processing, with a
reasonable size, to solve a practical clustering problem. You may select a dataset from one of the
following sources (other sources are also possible, e.g., Kaggle):
• UCI Machine Learning Repository, https://archive.ics.uci.edu/ml/datasets.php.
• KDD Cup challenges, http://www.kdd.org/kdd-cup
"""
logging.info(msg)

import gower
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle

from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.experimental import enable_iterative_imputer   ## <- NOTE required for IterativeImputer 
from sklearn.impute import IterativeImputer  ## <- NOTE experimental but better than SimpleImputer
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances, cosine_similarity
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from ucimlrepo import fetch_ucirepo


### - - - - - - - 
## https://archive.ics.uci.edu/dataset/342/mice+protein+expression
MICE_DATASET_ID = 342  ## Mice Protein Expression (80 x 1080) Data Set
MICE_DATASET_SHORT_NM = 'mice'
MICE_DATASET_NM = 'Mice Protein Expression'
MICE_SHORT_DESC = '''
Expression levels of 77 proteins measured in the cerebral cortex of 
8 classes of control and Down syndrome mice exposed to context fear 
conditioning, a task used to assess associative learning.
'''
## Mice paper: https://www.semanticscholar.org/paper/Self-Organizing-Feature-Maps-Identify-Proteins-to-a-Higuera-Gardiner/5c5754b02a4f2f36ccf8cdda78059cdb19860532

### - - - - - - - 
DATASET_ID = MICE_DATASET_ID
DATASET_SHORT_NM = MICE_DATASET_SHORT_NM
DATASET_NM = MICE_DATASET_NM

FETURES_FILE_NM = 'features.csv'
RAW_DATASET_FILE_NM = 'raw_dataset.pkl'
FETURES_FILE_NM = 'features.csv'

SCORE_SILHOUETTE = 'sil' ## Silhouette score
SCORE_DAVIES_BOULDIN = 'db'  ## Davies-Bouldin Index
SCORE_CALINSKI_HARABASZ = 'ch'  ## Calinski-Harabasz Index

##
##  Download and save if not found locally or load from local files
##  Return features and targets as X, y
##
def load_dataset(dataset_id:int=DATASET_ID, 
                raw_dataset_file:str=RAW_DATASET_FILE_NM,
                features_file:str=FETURES_FILE_NM,
                data_dir:str=DATA_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    
    raw_dataset_file = data_dir / f'{dataset_id}_{raw_dataset_file}'
    features_file = data_dir / f'{dataset_id}_{features_file}'
    if os.path.exists(features_file):
        logging.info(f'Loading dataset from local files: features {features_file}')
        data = pd.read_csv(features_file)
    else:
        logging.info(f'Downloading dataset {dataset_id} from UCI repository')
        dataset_raw = fetch_ucirepo(id=dataset_id)
        logging.info(f'loaded metadata: {dataset_raw.metadata}')
        logging.info(f'loaded variables: {dataset_raw.variables}')
        logging.info(f'saving raw dataset to a local file: {raw_dataset_file}')
        with open(raw_dataset_file, 'wb') as file:  ## TODO: make it conditional or remove
            pickle.dump(dataset_raw, file)
        data = dataset_raw.data.features
        logging.info(f'saving features and targets to local files: features {features_file}')
        data.to_csv(features_file, index=False)
    return data

##
## Handle missing values by predicting them
def pp_predict_missing(data:pd.DataFrame, columns: list)-> pd.DataFrame:
    imputer = IterativeImputer() ## <- NOTE! this is an experimental but better imputer than SimpleImputer
    data[columns] = imputer.fit_transform(data[columns])
    return data

##
## Handle missing values by removing ALL of them
def pp_remove_missing(data:pd.DataFrame)-> pd.DataFrame:
    data = data.dropna()
    return data

##
## Handle missing values by replacing them with the mean
def pp_mean_missing(data:pd.DataFrame, columns: list)-> pd.DataFrame:
    for col in columns:
        data[col] = data[col].fillna(data[col].mean())
    return data

##
## Handle missing values by replacing them with the median
def pp_median_missing(data:pd.DataFrame, columns: list)-> pd.DataFrame:
    for col in columns:
        data[col] = data[col].fillna(data[col].median())
    return data

##
## show missing ALL columns
def show_missing(dataset:pd.DataFrame)-> pd.Series:
    missing_values = dataset.isnull().sum()
    pd.set_option('display.max_rows', None) ## <--- Display ALL missing columns
    logging.debug(f"Missing values: {missing_values}")
    return missing_values

##
## this is more flexibe than pd.get_dummies, but we don't need it here
def pp_cat_to_onehot(data: pd.DataFrame, binary_columns: list[str]) -> pd.DataFrame:
    encoder = OneHotEncoder(drop='first', sparse=False)
    encoded_cols = encoder.fit_transform(data[binary_columns])
    encoded_df = pd.DataFrame(encoded_cols, columns=encoder.get_feature_names_out(binary_columns))
    data = data.drop(columns=binary_columns)
    data = pd.concat([data, encoded_df], axis=1)
    return data

##
##
def preprocess(dataset:pd.DataFrame, pp_file_path: str, force_pp:bool = False) -> pd.DataFrame:
    logging.info("Preprocessing: handling missing and categorical values...")

    if (not force_pp) and os.path.exists(pp_file_path):
        logging.info(f'Loading preprocessed dataset from local file: {pp_file_path} ....')
        dataset = pd.read_csv(pp_file_path)
        return dataset

    ## NOTE these are manually found and this is not the best way to do it, but will do for this assignment
    drop_columns = ['H3MeK4_N','BCL2_N','EGR1_N','BAD_N']
    predict_columns = ['H3AcK18_N','pCFOS_N']
    categorical_columns = ['Genotype', 'Treatment', 'Behavior']

    ## handle missing & categorical values
    logging.info(f"dropping columns {drop_columns} ...")
    dataset = dataset.drop(columns=drop_columns)
    continuous_columns = [col for col in dataset.columns if col not in categorical_columns]
    logging.debug(f"Predicting values for columns: {predict_columns}")
    dataset = pp_predict_missing(dataset, predict_columns) ## <- predict missing values using model
    ## NOTE: pd.get_dummies is easier to use and requires less code to implement than OneHotEncoder (see function above): 
    logging.debug(f"Converting categorical columns to one-hot using get_dummies and then numerical values: {categorical_columns}")
    dataset = pd.get_dummies(dataset, columns=categorical_columns, drop_first=True) ## <- convert categorical to one-hot
    logging.debug('Convert boolean columns to numerical values (0 and 1)...')
    boolean_columns = dataset.select_dtypes(include=['bool']).columns
    logging.debug(f"Converting boolean columns to numerical values: {boolean_columns}")
    dataset[boolean_columns] = dataset[boolean_columns].astype(int)
    logging.debug(f"Defaulting missing values using median in columns: {continuous_columns}")
    dataset = pp_median_missing(dataset, continuous_columns) ## <- replace missing values with median
    show_missing(dataset) ## <- NO missing values after preprocessing stage 1
    logging.info("Normalizing & Scaling data using MinMaxScaler...")
    dataset[continuous_columns] = MinMaxScaler().fit_transform(dataset[continuous_columns]) ## <- scale continuous features ONLY
    logging.debug(f"Dataset shape after pre-processing: {dataset.shape}")
    dataset.to_csv(pp_file_path, index=False) ## <----  save preprocessed dataset    
    return dataset

##
##
def pca_plot_2D(dist_matrix:np.ndarray, best_labels:np.ndarray, title: str, filename: str, show:bool=False)-> None:
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(dist_matrix)
    unique_labels = np.unique(best_labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    plt.figure(figsize=(10, 6))
    for label, color in zip(unique_labels, colors):
        if label == -1: ## <- noise points
            plt.scatter(pca_result[best_labels == label, 0], pca_result[best_labels == label, 1], 
                        c=[color], label='Noise', marker='o', s=3)
        else: ## <- cluster points
            plt.scatter(pca_result[best_labels == label, 0], pca_result[best_labels == label, 1], 
                        c=[color], label=f'Cluster {label}', marker='o', s=3)

    plt.title(title)
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend()
    plt.savefig(filename)
    if show: plt.show()

##
##
def pca_plot_3D(dist_matrix:np.ndarray, labels:np.ndarray, title: str, filename: str, show:bool=False)-> None:
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(dist_matrix)
    unique_labels = np.unique(labels)
    ## Manually define a list of contrasting colors to better visualize clusters
    colors = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', 
            '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', 
            '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', 
            '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080']
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    for label, color in zip(unique_labels, colors):
        if label == -1: ## <- noise points
            ax.scatter(pca_result[labels == label, 0], pca_result[labels == label, 1], pca_result[labels == label, 2], 
                    c=[color], label='Noise', marker='o', s=3)
        else: ## <- cluster points
            ax.scatter(pca_result[labels == label, 0], pca_result[labels == label, 1], pca_result[labels == label, 2], 
                    c=[color], label=f'Cluster {label}', marker='o', s=3)

    ax.set_title(title)
    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")
    ax.set_zlabel("PCA Component 3")
    ax.legend()
    plt.savefig(filename)
    if show: plt.show()

##
##
def plot_pca(dist_matrix, matrix_type, labels, cluster_type, title)-> None:
    pca_plot_2D(dist_matrix, labels, title, RESULTS_DIR / f'{DATASET_SHORT_NM}_{cluster_type}_{matrix_type}_clstrs_pca_2D.png')
    pca_plot_3D(dist_matrix, labels, title, RESULTS_DIR / f'{DATASET_SHORT_NM}_{cluster_type}_{matrix_type}_clstrs_pca_3D.png')

##
##
def plot_cluster_sizes(cluster_sizes, title, name, show:bool=False)-> None:
    plt.figure(figsize=(10, 6))
    cluster_sizes.sort_index().plot(kind='bar')
    plt.title(title)
    plt.xlabel("Cluster Label")
    plt.ylabel("Number of Instances")
    plt.savefig(RESULTS_DIR / f'{DATASET_SHORT_NM}_{name}_clstrs_sizes.png')
    if show: plt.show()

##
## 
def cluster_kmeans(data, k, dist_matrix, matrix_type:str) -> dict[str, float]:
    kmeans = KMeans(n_clusters=k, random_state=42, init='k-means++', n_init='auto', max_iter=600)
    kmeans.fit(dist_matrix)
    data['cluster'] = kmeans.labels_
    cluster_sizes = data['cluster'].value_counts()
    logging.debug(f"KMeans cluster sizes using {matrix_type} with k={k}:\n{cluster_sizes}")
    logging.info(f"Visualize KMeans clustering results using {matrix_type} and k={k}")
    plot_cluster_sizes(cluster_sizes, f"KMeans Clustering using {matrix_type} with k={k}", f'kmeans_{matrix_type}_{k}')
    plot_pca(dist_matrix, matrix_type, kmeans.labels_,f'kmenas_{k}',f"KMeans Clustering using {matrix_type} with k={k}")
    return calc_score(f'KMeans using {matrix_type} with k={k}', dist_matrix, kmeans.labels_)

##
##
def cluster_agg(data, dist_matrix, matrix_type, k)-> dict[str, float]:
    clustering = AgglomerativeClustering(n_clusters=k, metric='precomputed', linkage='average')
    cluster_labels = clustering.fit_predict(dist_matrix)
    data['agg_cluster'] = cluster_labels
    cluster_sizes = data['agg_cluster'].value_counts() 
    logging.debug(f"Agglomerative cluster sizes using {matrix_type} and k={k}:\n{cluster_sizes}")
    logging.info(f"Visualize AgglomerativeClustering results using {matrix_type} and k={k}")
    plot_cluster_sizes(cluster_sizes, f"Agglomerative Clustering using {matrix_type} with k={k}", f'agg_{matrix_type}_{k}')
    plot_pca(dist_matrix, matrix_type, cluster_labels, f'agg_{k}', f"Agglomerative Clustering using {matrix_type} with k={k}")
    return calc_score(f'AgglomerativeClustering using {matrix_type} with k={k}', dist_matrix, cluster_labels)
    

##
##
def cluster_dbscan(data, dist_matrix, matrix_type, eps, min_samples)-> dict[str, float]:
    best_eps = None
    best_min_samples = None
    best_noise_count = float('inf')
    best_labels = None
    for eps in eps:
        for min_sample in min_samples:
            dbscan = DBSCAN(eps=eps, min_samples=min_sample)
            dbscan.fit(dist_matrix)
            labels = dbscan.labels_
            noise_count = list(labels).count(-1)
            logging.debug(f"DBSCAN using {matrix_type}, eps={eps}, min_samples={min_sample} noise count: {noise_count}")
            if noise_count < best_noise_count:
                best_eps = eps
                best_min_samples = min_sample
                best_noise_count = noise_count
                best_labels = labels
 
    data['dbscan_cluster'] = best_labels
    cluster_sizes = data['dbscan_cluster'].value_counts()
    logging.info(f"Best DBSCAN using {matrix_type}, eps={best_eps}, min_samples={best_min_samples} with noise count: {best_noise_count}")
    logging.info(f"DBSCAN using {matrix_type}, eps={best_eps}, min_samples={best_min_samples} cluster sizes:\n{cluster_sizes}")
    logging.info(f"Visualize DBSCAN clustering results using {matrix_type} for best eps={best_eps}, min_samples={best_min_samples}")
    plot_cluster_sizes(cluster_sizes, f"DBSCAN Clustering using {matrix_type} with eps={best_eps}, min_samples={best_min_samples}", f'dbscan_{matrix_type}_{best_eps}_{best_min_samples}')
    plot_pca(dist_matrix, matrix_type, best_labels, 'dbscan', f"DBSCAN Clustering using {matrix_type} with eps={best_eps}, min_samples={best_min_samples}")
    return calc_score(f'DBSCAN using {matrix_type} with eps={best_eps}, min_samples={best_min_samples}', dist_matrix, best_labels)


##
##
def calc_score(clust_type, dist_matrix, labels) -> dict[str, float]:
    scores = {}
    try:
        scores[SCORE_SILHOUETTE] = silhouette_score(dist_matrix, labels)
    except Exception as e:
        scores[SCORE_SILHOUETTE] = None
        logging.error(f"Error calculating {SCORE_SILHOUETTE} for {clust_type}: {e}")
    try:
        scores[SCORE_DAVIES_BOULDIN] = davies_bouldin_score(dist_matrix, labels)
    except Exception as e:
        scores[SCORE_DAVIES_BOULDIN] = None
        logging.error(f"Error calculating {SCORE_DAVIES_BOULDIN} for {clust_type}: {e}")
    try:
        scores[SCORE_CALINSKI_HARABASZ] = calinski_harabasz_score(dist_matrix, labels)
    except Exception as e:
        scores[SCORE_CALINSKI_HARABASZ] = None
        logging.error(f"Error calculating {SCORE_CALINSKI_HARABASZ} for {clust_type}: {e}")
    
    logging.info(f"{clust_type} - Silhouette Score: {scores[SCORE_SILHOUETTE]}")
    logging.info(f"{clust_type} - Davies-Bouldin Index: {scores[SCORE_DAVIES_BOULDIN]}")
    logging.info(f"{clust_type} - Calinski-Harabasz Index: {scores[SCORE_CALINSKI_HARABASZ]}")

    return scores

##
##
def get_data(pp_file_path, force_preprocessing) -> pd.DataFrame:
    if (not force_preprocessing) and os.path.exists(pp_file_path): 
        logging.info(f'Loading already preprocessed dataset from local file: {pp_file_path} ....')
        data = pd.read_csv(pp_file_path)
    else:
        logging.info("Loading the dataset...")
        data = load_dataset(DATASET_ID)
        logging.info("Analyzing the dataset...")
        logging.debug(f"Dataset initial shape: {data.shape}")
        show_missing(data) ## <- show missing values is just one step
        logging.info("Preprocessing...")
        data = preprocess(data, pp_file_path, force_preprocessing)
    return data

##
##
def get_matrix_dict(data) -> dict[str, np.ndarray]:
    dist_matrices = {}
    dist_matrices['Gower'] = gower.gower_matrix(data)
    dist_matrices['Euclidean'] = euclidean_distances(data)
    dist_matrices['Manhattan'] = manhattan_distances(data)
    dist_matrices['Cosine'] = 1 - cosine_similarity(data)
    return dist_matrices

##
##
def cluster(data, dist_matrices, k_list:list) -> dict[str, float]:
    scores = {}
    for matrix_type, dist_matrix in dist_matrices.items():
        for k in k_list:
            logging.info(f"KMeans clustering using {matrix_type} and cluster size {k}...")
            scores[f'{matrix_type}_{k}_kmenas'] = cluster_kmeans(data, k, dist_matrix, matrix_type)
            logging.info(f"AgglomerativeClustering using {matrix_type} and cluster size {k}...")
            scores[f'{matrix_type}_{k}_agg'] = cluster_agg(data, dist_matrix, matrix_type, k)
        logging.info(f"DBSCAN clustering using {matrix_type}...")
        eps = [0.2, 0.4, 0.6]
        min_samples = [5, 10, 15]
        scores[f'{matrix_type}_{k}_dbscan'] = cluster_dbscan(data, dist_matrix, matrix_type, eps, min_samples)
    logging.info("Clustering and visualization completed.")
    return scores

##
##
def process_socores(scores) -> pd.DataFrame:
    results = pd.DataFrame.from_dict(scores, orient='index') ## inner keys are scores they are columns
    ## Calculate distances from the best possible score for each metric
    results['sil_dist'] =  np.abs(results[SCORE_SILHOUETTE] - 1)    ## <- closest to 1 is the best
    results['db_dist'] =  np.abs(results[SCORE_DAVIES_BOULDIN] - 0) ## <- closest to zero is the best
    results['ch_dist'] =  -results[SCORE_CALINSKI_HARABASZ]         ## <- closest to the positive infinity is the best
    ## Normalize the distances using min-max scaling
    results['sil_dist'] = (results['sil_dist'] - results['sil_dist'].min()) / (results['sil_dist'].max() - results['sil_dist'].min())
    results['db_dist'] = (results['db_dist'] - results['db_dist'].min()) / (results['db_dist'].max() - results['db_dist'].min())
    results['ch_dist'] = (results['ch_dist'] - results['ch_dist'].min()) / (results['ch_dist'].max() - results['ch_dist'].min())
    ## Calculate the total distance for each row to find the best overall clustering result
    results['total_dist'] = results['sil_dist'] + results['db_dist'] + results['ch_dist']
    results.to_csv(RESULTS_DIR / f'{DATASET_SHORT_NM}_clustering_scores.csv', index=False)
    logging.info("Results saved to file.")
    return results

##
##
def save_best_results(results) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]: 
    best_overall = results.loc[results['total_dist'].idxmin()] 
    best_sil = results.loc[results['sil_dist'].idxmin()]
    best_db = results.loc[results['db_dist'].idxmin()]
    best_ch = results.loc[results['ch_dist'].idxmin()]
    logging.info(f"Row with the best overall clustering result:\n{best_overall}")
    logging.info(f"Row with the best Silhouette Score:\n{best_sil}")
    logging.info(f"Row with the best Davies-Bouldin Index:\n{best_db}")
    logging.info(f"Row with the best Calinski-Harabasz Index:\n{best_ch}")
    best_results = pd.DataFrame([best_overall, best_sil, best_db, best_ch])
    best_results.index = ['Best Overall', 'Best Silhouette', 'Best Davies-Bouldin', 'Best Calinski-Harabasz']
    best_results['Model'] = best_results.index.map(lambda x:  ## Add the model name (from results df) as a column
                                results.loc[results['total_dist'].idxmin()].name if x == 'Best Overall' else
                                results.loc[results['sil_dist'].idxmin()].name if x == 'Best Silhouette' else
                                results.loc[results['db_dist'].idxmin()].name if x == 'Best Davies-Bouldin' else
                                results.loc[results['ch_dist'].idxmin()].name)
    best_results.reset_index(inplace=True)  ## Reset the index and name it Metric
    best_results.rename(columns={'index': 'Metric'}, inplace=True)
    best_results.to_csv(RESULTS_DIR / f'{DATASET_SHORT_NM}_best_clustering_results.csv', index=True)
    return best_overall, best_sil, best_db, best_ch



###
#### -----------------------------------
###
if __name__ == '__main__':
    logging.info("Starting the script...")
    pp_file_path = DATA_DIR / f'{DATASET_ID}_preproc_data.csv'
    force_preprocessing = False
    logging.info("Load and preprocess dataset...")
    data = get_data(pp_file_path, force_preprocessing)
    logging.info("Calculate distance matrices...")
    dist_matrices = get_matrix_dict(data)
    logging.info("Cluster and visualize...")
    scores = cluster(data, dist_matrices, k_list=[4,6,8])
    results = process_socores(scores)
    logging.info(f"Results:\n{results}")
    best_results = save_best_results(results)
    logging.info(f"Best results:\n{best_results}")
    logging.info("Script completed.")
