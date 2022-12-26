import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn import set_config
import scipy.stats as ss
import itertools
import pandas as pd



def hist_distributions(dataframe, figsize:tuple=(20,10), rows:int=3, cols:int=4):
    plt.figure(figsize=figsize)
    for index, column in enumerate(dataframe.columns):
        plt.subplot(rows,cols,index+1)
        plt.hist(dataframe[column])
        plt.title(f"Distribution by : {column}", fontsize=15)
        if rows[0] == str(""):
            plt.tick_params(axis=x, bottom='False')
    plt.show()


def scatter_figure_byTarget(dataframe, target:str, figsize:tuple=(20,10), rows:int=3, cols:int=4):
    dataframedrop = dataframe.drop(columns=target)
    plt.figure(figsize=figsize)
    for index, column in enumerate(dataframedrop.columns):
        plt.subplot(rows,cols,index+1)
        plt.scatter(dataframedrop[column], dataframe[target])
        plt.title(f'{target}/{column}', fontsize=15)
    plt.show()


def make_heatmap(dataframe):
    heatmap_df = dataframe.corr()
    plt.figure(figsize=(15,12))
    sns.heatmap(heatmap_df, annot=True)
    plt.show()


def make_mix_pipeline(model=LinearRegression(), cat_transformer=OneHotEncoder(handle_unknown='ignore'), num_encoder=MinMaxScaler()):
    num_transformer = make_pipeline(SimpleImputer(strategy='median'),num_encoder)
    numerical_features = make_column_selector(dtype_include = np.number)
    cat_features = make_column_selector(dtype_exclude = np.number)
    preprocessing_transformer = ColumnTransformer([('num columns',num_transformer, numerical_features),('cat columns',cat_transformer, cat_features)])
    pipeline_workflow = make_pipeline(preprocessing_transformer, model)
    return pipeline_workflow


def view_unique_values(dataframe, count:bool=True):
        if count:
            [print(f'column {index} : {column} => {len(dataframe[column].unique())} uniques values') for index, column in enumerate(dataframe.columns)]
        else:
            [print(f'column {index} : {column}\n{dataframe[column].unique()}\n\n') for index, column in enumerate(dataframe.columns)]


def cramers_v_matrix(dataframe, variables):
    df = pd.DataFrame(index=dataframe[variables].columns,
                      columns=dataframe[variables].columns,
                      dtype="float64")

    for v1, v2 in itertools.combinations(variables, 2):

        # generate contingency table:
        table = pd.crosstab(dataframe[v1], dataframe[v2])
        n     = len(dataframe.index)
        r, k  = table.shape

        # calculate chi squared and phi
        chi2  = ss.chi2_contingency(table)[0]
        phi2  = chi2/n

        # bias corrections:
        r = r - ((r - 1)**2)/(n - 1)
        k = k - ((k - 1)**2)/(n - 1)
        phi2 = max(0, phi2 - (k - 1)*(r - 1)/(n - 1))

        # fill correlation matrix
        df.loc[v1, v2] = np.sqrt(phi2/min(k - 1, r - 1))
        df.loc[v2, v1] = np.sqrt(phi2/min(k - 1, r - 1))
        np.fill_diagonal(df.values, np.ones(len(df)))
    return df


def make_cat_heatmap(dataframe, cat_column:list):
    sns.heatmap(cramers_v_matrix(dataframe, cat_column), annot=True)
    plt.show()


def view_values_frequency(serie:list, graphic_mode:bool=False):
    if graphic_mode:
        plt.figure(figsize=(20,10))
        plt.hist(serie, color='green', bins=len(serie.unique()))
        plt.xlabel(f'Distribution by "{serie.name}"', fontsize=15)
        plt.show()
        return None
    else:
        return (serie.value_counts()/serie.value_counts().sum())*100

def calc_outliers_std(data, sigma_mul=3):
    mean = np.mean(data)
    std = np.std(data)
    outliers = data[np.abs(data - mean) > sigma_mul * std]
    return outliers

def calc_outliers(data, q1=0.25, q3=0.75):
    # Calculate the first and third quartiles and the IQR
    Q1 = data.quantile(q1)
    Q3 = data.quantile(q3)
    IQR = Q3 - Q1

    # Define the range of acceptable values
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Identify outliers
    outliers = data[(data < lower_bound) | (data > upper_bound)]

    return outliers