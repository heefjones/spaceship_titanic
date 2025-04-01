# data science
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy.stats import normaltest

# machine learning
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import log_loss, accuracy_score
from bayes_opt import BayesianOptimization

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# display
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
sns.set(style='whitegrid', font='Average')

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# global vars
ROOT = './data/'

# set numpy seed
SEED = 9
np.random.seed(SEED)

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def show_shape_and_nulls(df):
    """
    Display the shape of a DataFrame and the number of null values in each column.

    Args:
    - df (pd.DataFrame): The DataFrame to analyze.

    Returns:
    - None
    """

    # print shape
    print(f'Shape: {df.shape}')

    # check for missing values
    print('Null values:')

    # display null values
    display(df.isnull().sum().to_frame().T)

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def show_unique_vals(df):
    """
    Print the number of unique values for each column in a DataFrame.
    If a column has fewer than 20 unique values, print those values.

    Args:
    - df (pd.DataFrame): The DataFrame to analyze.

    Returns:
    - None
    """

    # iterate over columns
    for col in df.columns:
        # get number of unique values and print
        n = df[col].nunique()
        print(f'"{col}" has {n} unique values')

        # if number of unique values is under 20, print the unique values
        if n < 20:
            print(df[col].unique())
        print()

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def create_feature_sets(df, degree=2):
    """
    Generate multiple feature sets by applying polynomial, interaction, and custom transformations.
    
    Args:
    - df (pd.DataFrame): Original DataFrame with preprocessed columns.
    - degree (int): Maximum polynomial degree to generate. Default is 2.
    
    Returns:
    - feature_sets (dict): Dictionary of feature sets with variations.
    """
    feature_sets = {}

    # list of numeric and categorical columns
    num_cols = df.select_dtypes(include=['number']).columns

    # base features
    feature_sets['base_features'] = df.copy()

    # polynomial features
    poly = PolynomialFeatures(degree=degree, interaction_only=False, include_bias=False)
    poly_features = poly.fit_transform(df[num_cols])
    poly_df = pd.DataFrame(poly_features, columns=poly.get_feature_names_out(num_cols))
    poly_df.index = df.index
    feature_sets['poly_features'] = pd.concat([df.drop(num_cols, axis=1), poly_df], axis=1)

    # polynomial features (doubled)
    poly_doubled = PolynomialFeatures(degree=degree*2, interaction_only=False, include_bias=False)
    poly_features_doubled = poly_doubled.fit_transform(df[num_cols])
    poly_doubled_df = pd.DataFrame(poly_features_doubled, columns=poly_doubled.get_feature_names_out(num_cols))
    poly_doubled_df.index = df.index
    feature_sets['poly_doubled_features'] = pd.concat([df.drop(num_cols, axis=1), poly_doubled_df], axis=1)

    # interaction features (only pairwise)
    interaction = PolynomialFeatures(degree=degree, interaction_only=True, include_bias=False)
    interaction_features = interaction.fit_transform(df[num_cols])
    interaction_df = pd.DataFrame(interaction_features, columns=interaction.get_feature_names_out(num_cols))
    interaction_df.index = df.index
    feature_sets['interaction_features'] = pd.concat([df.drop(num_cols, axis=1), interaction_df], axis=1)

    # polynomial + interaction features
    poly_interaction_features = interaction.fit_transform(poly_df)
    poly_interaction_df = pd.DataFrame(poly_interaction_features, columns=interaction.get_feature_names_out(poly_df.columns))
    poly_interaction_df.index = df.index
    poly_interaction_df = poly_interaction_df.T.drop_duplicates().T
    feature_sets['poly_interaction_features'] = pd.concat([df.drop(num_cols, axis=1), poly_interaction_df], axis=1)
    
    return feature_sets

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def cross_val_model(estimator, data_name, data, scaler, models_df, folds=10):
    """
    Perform KFold cross validation on a given estimator and store evaluation metrics.
    
    Args:
    - estimator (sklearn estimator): Estimator to use for modeling.
    - data_name (str): Name of the dataset being used.
    - data (pd.DataFrame): Data to model.
    - scaler (sklearn scaler, optional): Scaler to use for data.
    - models_df (pd.DataFrame): DataFrame to save model results to. Expected columns: ['Model', 'Scaler', 'Feature_Set', 'Train_LogLoss', 'Val_LogLoss', 'Train_Acc', 'Val_Acc']
    - folds (int): Number of cross-validation folds to use. Default is 10.
    
    Returns:
    - models_df (pd.DataFrame): Updated DataFrame with a new row containing model evaluation metrics.
    """
    
    # define numerical and categorical selectors
    num_selector = make_column_selector(dtype_include='number')
    cat_selector = make_column_selector(dtype_exclude='number')

    # create column transformer to handle encoding
    preprocessor = make_column_transformer((OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_selector), remainder='passthrough')

    # define the pipeline
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('scaler', scaler), ('model', estimator)])

    # define X and y
    X = data.drop(['PassengerId', 'Transported'], axis=1)
    y = data['Transported']

    # init lists to store metrics across folds
    train_logloss_list, val_logloss_list = [], []
    train_acc_list, val_acc_list = [], []
    
    # iterate through k-folds
    kf = KFold(n_splits=folds, shuffle=True, random_state=SEED)
    for train_index, val_index in kf.split(X):
        # split X and y
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]
        
        # fit the pipeline
        pipeline.fit(X_train, y_train)
        
        # predict on train and validation sets
        y_train_pred = pipeline.predict(X_train)
        y_val_pred = pipeline.predict(X_val)
        
        # log loss
        train_loss = log_loss(y_train, y_train_pred)
        val_loss = log_loss(y_val, y_val_pred)

        # calculate accuracy
        train_acc = accuracy_score(y_train, y_train_pred)
        val_acc = accuracy_score(y_val, y_val_pred)
        
        # append metrics for this fold
        train_logloss_list.append(train_loss)
        val_logloss_list.append(val_loss)
        train_acc_list.append(train_acc)
        val_acc_list.append(val_acc)
    
    # prepare a new row with averages
    new_row = {
        'Model': estimator.__class__.__name__,
        'Scaler': scaler.__class__.__name__,
        'Feature_Set': data_name,
        'Train_LogLoss': np.mean(train_logloss_list),
        'Val_LogLoss': np.mean(val_logloss_list),
        'Train_Acc': np.mean(train_acc_list),
        'Val_Acc': np.mean(val_acc_list)
    }

    # append the new row
    models_df.loc[len(models_df)] = new_row

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def make_pipeline(model, scaler):
    """
    Create a pipeline with column preprocessing, a scaler, and a model.

    Args:
    - model: a machine learning model
    - scaler: a scaler object

    Returns:
    - pipeline: a pipeline object
    """
    
    # define numerical and categorical selectors
    num_selector = make_column_selector(dtype_include='number')
    cat_selector = make_column_selector(dtype_exclude='number')

    # create column transformer to handle encoding and scaling
    preprocessor = make_column_transformer((OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_selector), remainder='passthrough')

    # define a pipeline with preprocessing, scaling, and model
    pipeline = Pipeline([('preprocessor', preprocessor), ('scaler', scaler), ('model', model)])

    return pipeline

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def xgb_cv(max_depth, n_estimators, learning_rate, gamma, min_child_weight, subsample, colsample_bytree, colsample_bylevel, colsample_bynode, X, y):
    """
    Objective function for XGBoost hyperparameter tuning using Bayesian Optimization.

    Args:
    - XGBClassifier parameters: max_depth, n_estimators, learning_rate, gamma, min_child_weight, subsample, colsample_bytree, colsample_bylevel, colsample_bynode
    - X (pd.DataFrame): Feature set.
    - y (pd.Series): Target variable.

    Returns:
    - scores.mean() (float): Mean log loss from 10-fold cross-validation.
    """

    # define XGBoost parameters
    params = {'max_depth': int(max_depth),
        'n_estimators': int(n_estimators),
        'learning_rate': learning_rate,
        'gamma': gamma,
        'min_child_weight': min_child_weight,
        'subsample': subsample,
        'colsample_bytree': colsample_bytree,
        'colsample_bylevel': colsample_bylevel,
        'colsample_bynode': colsample_bynode,
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'use_label_encoder': False,
        'device': 'cuda',
        'tree_method': 'hist',
        'random_state': SEED}

    # create pipeline
    pipeline = make_pipeline(XGBClassifier(**params), StandardScaler())

    # 10-fold cross-validation
    kf = KFold(n_splits=10, shuffle=True, random_state=SEED)
    scores = cross_val_score(pipeline, X, y, cv=kf, scoring='neg_log_loss')

    # return mean cv score
    return scores.mean()