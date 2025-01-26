"""Main module to process the Data from the AmsterdamUMCDB.

Author: Mika Florin Rosin
"""
import os
import json

import pandas as pd
import pandas_gbq as pd_gbq
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import boxcox, skew, yeojohnson
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from google.cloud.bigquery.client import Client
from google.cloud import bigquery

from src.data_processing.queries.combined_diagnosis import combined_diagnoses_query
from src.data_processing.queries.sofa_scores import *
from src.data_processing.queries.data_queries import query_demographics, query_measurement, query_ventilation
from src.data_processing.missing_data import plot_missing_data, calculate_average_nans, drop_na_cols
from src.data_processing.data_imputation import (
    calculate_average_nans,
    knn_impute_by_column,
    forward_fill_by_column,
    forward_fill_limited,
    drop_features_with_missing_data,
    knn_impute_dataset,
    KNNImputer
)
from src.clustering import cluster_kmpp, evaluate_clustering
from src.data_processing.outliers import drop_outliers_iqr_long, drop_outliers_iqr_wide

from src.dkm import Autoencoder, DKNLoss, training, DataLoader
from src.dataset import AmsICUSepticShock
import torch.optim as optim
import torch
from torch.utils.data import Dataset, DataLoader

tqdm.pandas()


def get_sofa_resp(df_sofa_resp):
    #remove extreme outliers
    df_sofa_resp.loc[(df_sofa_resp['fio2'] > 100), 'fio2'] = np.nan
    #convert FiO2 in % to fraction
    df_sofa_resp.loc[(df_sofa_resp['fio2'] <= 100) & (df_sofa_resp['fio2'] >= 20) , 'fio2'] = df_sofa_resp['fio2']/100
    #remove extreme outliers (FiO2) (possible O2 flow?)
    df_sofa_resp.loc[(df_sofa_resp['fio2'] > 1), 'fio2'] = np.nan
    #remove lower outliers, most likely incorrectly labeled as 'arterial' instead of '(mixed/central) venous'
    df_sofa_resp['pao2'] = df_sofa_resp['pao2'].astype(float)
    df_sofa_resp['fio2'] = df_sofa_resp['fio2'].astype(float)
    df_sofa_resp.loc[df_sofa_resp['pao2'] < 50, 'pao2'] = np.nan
    df_sofa_resp = df_sofa_resp.dropna(subset=['pao2']).copy()
    df_sofa_resp['pf_ratio'] = df_sofa_resp['pao2'] / df_sofa_resp['fio2']

    #calculate SOFA respiration score:
    df_sofa_resp.loc[:,'sofa_respiration_score'] = 0
    df_sofa_resp.loc[
        (df_sofa_resp['pf_ratio'] < 400) & (df_sofa_resp['pf_ratio'] >= 300),
        'sofa_respiration_score'
    ] = 1
    df_sofa_resp.loc[
        (df_sofa_resp['pf_ratio'] < 300),
        'sofa_respiration_score'
    ] = 2
    df_sofa_resp.loc[
        (df_sofa_resp['pf_ratio'] < 200) &
        (df_sofa_resp['pf_ratio'] >= 100) &
        (df_sofa_resp['ventilatory_support'] == True),
        'sofa_respiration_score'
    ] = 3
    df_sofa_resp.loc[
        (df_sofa_resp['pf_ratio'] < 100) & (df_sofa_resp['ventilatory_support'] == True),
        'sofa_respiration_score'
    ] = 4

    return df_sofa_resp


def get_sofa_coagulation(sofa_coagulation):
        #calculate SOFA coagulation score:
    sofa_coagulation.loc[:,'sofa_coagulation_score'] = 0
    sofa_coagulation.loc[
        (sofa_coagulation['value'] < 150) & (sofa_coagulation['value'] >= 100),
        'sofa_coagulation_score'
    ] = 1
    sofa_coagulation.loc[
        (sofa_coagulation['value'] < 100) & (sofa_coagulation['value'] >= 50),
        'sofa_coagulation_score'
    ] = 2
    sofa_coagulation.loc[
        (sofa_coagulation['value'] < 50) & (sofa_coagulation['value'] >= 20),
        'sofa_coagulation_score'
    ] = 3
    sofa_coagulation.loc[
        (sofa_coagulation['value'] < 20),
        'sofa_coagulation_score'
    ] = 4

    return sofa_coagulation


def get_sofa_liver(sofa_liver):
    #calculate SOFA liver score:
    sofa_liver.loc[:,'sofa_liver_score'] = 0
    sofa_liver.loc[
        (sofa_liver['value'] >= 20) & (sofa_liver['value'] < 33),
        'sofa_liver_score'
    ] = 1
    sofa_liver.loc[
        (sofa_liver['value'] >= 33) & (sofa_liver['value'] < 102),
        'sofa_liver_score'
    ] = 2
    sofa_liver.loc[
        (sofa_liver['value'] >= 102) & (sofa_liver['value'] < 204),
        'sofa_liver_score'
    ] = 3
    sofa_liver.loc[(sofa_liver['value'] >= 204), 'sofa_liver_score'] = 4
    return sofa_liver


def get_cardiovascular_meds(sofa_cardiovascular_meds):
    sofa_cardiovascular_meds = sofa_cardiovascular_meds.groupby(['visit_occurrence_id','itemid', 'item']).agg(
        total_duration=pd.NamedAgg(column='duration', aggfunc='sum'),
        max_gamma=pd.NamedAgg(column='gamma', aggfunc='max')
    ).reset_index()

        #calculate SOFA cardiovascular score:
    sofa_cardiovascular_meds.loc[:,'sofa_cardiovascular_score'] = 0

    #dopamine (itemid 36411287) <= 5 or dobutamine (itemid 21088391) any dose
    sofa_cardiovascular_meds.loc[(
        ((sofa_cardiovascular_meds['itemid'] == 36411287) & (sofa_cardiovascular_meds['max_gamma'] <= 5)) |
        ((sofa_cardiovascular_meds['itemid'] == 21088391))
    ), 'sofa_cardiovascular_score'] = 2

    #dopamine (itemid 36411287) > 5, epinephrine (itemid 19076867) <= 0.1, norepinephrine (itemid 2907531) <= 0.1
    sofa_cardiovascular_meds.loc[(
        ((sofa_cardiovascular_meds['itemid'] == 36411287.0) & (sofa_cardiovascular_meds['max_gamma'] > 5) &
        (sofa_cardiovascular_meds['max_gamma'] < 15)) |
        ((sofa_cardiovascular_meds['itemid'] == 19076867.0) & (sofa_cardiovascular_meds['max_gamma'] <= 0.1)) |
        ((sofa_cardiovascular_meds['itemid'] == 2907531.0) & (sofa_cardiovascular_meds['max_gamma'] <= 0.1))
    ), 'sofa_cardiovascular_score'] = 3

    #dopamine (itemid 36411287) > 15, epinephrine (itemid 19076867) > 0.1, norepinephrine (itemid 2907531) > 0.1

    sofa_cardiovascular_meds.loc[(
        ((sofa_cardiovascular_meds['itemid'] == 36411287) & (sofa_cardiovascular_meds['max_gamma'] > 15)) |
        ((sofa_cardiovascular_meds['itemid'] == 19076867) & (sofa_cardiovascular_meds['max_gamma'] > 0.1)) |
        ((sofa_cardiovascular_meds['itemid'] == 2907531) & (sofa_cardiovascular_meds['max_gamma'] > 0.1))
    ), 'sofa_cardiovascular_score'] = 4

    return sofa_cardiovascular_meds


def get_blood_pressure(mean_abp):
    mean_abp.loc[(mean_abp['value'] > 165), 'value'] = np.nan
    mean_abp.loc[(mean_abp['value'] <= 30), 'value'] = np.nan
    mean_abp = mean_abp.dropna()
    sofa_cardiovascular_map = mean_abp.groupby(['visit_occurrence_id', 'itemid', 'item']).agg(
        lowest_mean_abp=pd.NamedAgg(column='value', aggfunc='min')
    ).reset_index()
    #calculate SOFA cardiovascular score:
    sofa_cardiovascular_map.loc[:,'sofa_cardiovascular_score'] = 0
    #MAP < 70
    sofa_cardiovascular_map.loc[(sofa_cardiovascular_map['lowest_mean_abp'] < 70), 'sofa_cardiovascular_score'] = 1

    return sofa_cardiovascular_map


def get_sofa_cns(sofa_cns):
    sofa_cns = sofa_cns.groupby(['visit_occurrence_id']).agg(
        min_gcs=pd.NamedAgg(column='gcs_score', aggfunc='min')
    ).reset_index()

    #calculate SOFA Central nervous system score:
    sofa_cns.loc[:,'sofa_cns_score'] = 0
    sofa_cns.loc[(sofa_cns['min_gcs'] >= 13) & (sofa_cns['min_gcs'] < 15), 'sofa_cns_score'] = 1
    sofa_cns.loc[(sofa_cns['min_gcs'] >= 10) & (sofa_cns['min_gcs'] < 13), 'sofa_cns_score'] = 2
    sofa_cns.loc[(sofa_cns['min_gcs'] >= 6) & (sofa_cns['min_gcs'] < 10), 'sofa_cns_score'] = 3
    sofa_cns.loc[(sofa_cns['min_gcs'] < 6), 'sofa_cns_score'] = 4

    return sofa_cns


def get_sofa_renal(sofa_renal_urine_output):
    #probably decimal error when entering volumes > 2500
    sofa_renal_urine_output.loc[
        (sofa_renal_urine_output['value'] > 2500),
        'value'
    ] = sofa_renal_urine_output['value']/10
    #remove extreme outliers, most likely data entry error)
    sofa_renal_urine_output.loc[
        (sofa_renal_urine_output['value'] > 4500),
        'value'
    ] = np.nan
    sofa_renal_urine_output = sofa_renal_urine_output.dropna()
    #get urine output per 24 hours
    sofa_renal_daily_urine_output = sofa_renal_urine_output.groupby(['visit_occurrence_id']).agg(
        daily_urine_output=pd.NamedAgg(column='value', aggfunc='sum')
    ).reset_index()
    sofa_renal_daily_urine_output.head()
    #calculate SOFA renal score for urine output:
    sofa_renal_daily_urine_output.loc[:,'sofa_renal_score'] = 0

    #urine output < 500 ml/day
    sofa_renal_daily_urine_output.loc[(
        ((sofa_renal_daily_urine_output['daily_urine_output'] < 500) &
        (sofa_renal_daily_urine_output['daily_urine_output'] > 200))),
        'sofa_renal_score'
    ] = 3

    #urine output < 200 ml/day
    sofa_renal_daily_urine_output.loc[(
        ((sofa_renal_daily_urine_output['daily_urine_output'] < 200))),
        'sofa_renal_score'
    ] = 4

    return sofa_renal_daily_urine_output


def get_initial_septic_shock(df_lactate):
    #df_lactate.loc[
    #    (df_lactate['value'] > 20),
    #    'value'
    #] = np.nan
    df_lactate = df_lactate.dropna()
    #get max lactate over 24 hours
    df_lactate = df_lactate.groupby(['visit_occurrence_id']).agg(
        max_lactate=pd.NamedAgg(column='value', aggfunc='max')
    ).reset_index()
    # shock for patients
    df_lactate.loc[:,'septic_shock'] = False

    #urine output < 200 ml/day
    df_lactate.loc[(
        ((df_lactate['max_lactate'] > 2.0))),
        'septic_shock'
    ] = True

    return df_lactate


def get_sofa_creatinine(df_creatinine):
    #looking at the data it's relevatively easy to spot most lab collection errors (i.e. single outliers between relatively
    # normal values
    # TO DO: algorithm to remove these errors, but not 'real' outliers
    df_creatinine.loc[
        df_creatinine['visit_occurrence_id'].isin(
            df_creatinine.loc[
                (df_creatinine['value']> 1000),
                'visit_occurrence_id'
            ])
    ].sort_values(by='visit_occurrence_id')
    df_creatinine.loc[
        df_creatinine['visit_occurrence_id'].isin(
            df_creatinine.loc[(df_creatinine['value'] < 30), 'visit_occurrence_id']
        )
    ].sort_values(by='visit_occurrence_id')
    #remove extreme outliers, most likely data entry errors (manual_entry = True)
    df_creatinine.loc[(df_creatinine['value'] < 30), 'value'] = np.nan
    df_creatinine = df_creatinine.dropna(subset=['value'])
    #get highest creatinine per 24 hours
    #use creatinine 'cleansed' dataframe from APACHE score
    sofa_renal_creatinine = df_creatinine.groupby(['visit_occurrence_id']).agg(
            max_creatinine=pd.NamedAgg(column='value', aggfunc='max')
        ).reset_index()
    #calculate SOFA renal score for creatinine:
    sofa_renal_creatinine.loc[:,'sofa_renal_score'] = 0

    #creatinine 110-170 umol/l
    sofa_renal_creatinine.loc[(
        ((sofa_renal_creatinine['max_creatinine'] >= 110) &
        (sofa_renal_creatinine['max_creatinine'] < 171))
    ), 'sofa_renal_score'] = 1

    #creatinine 171-299 umol/l
    sofa_renal_creatinine.loc[(
        ((sofa_renal_creatinine['max_creatinine'] >= 171) &
        (sofa_renal_creatinine['max_creatinine'] < 300))
    ), 'sofa_renal_score'] = 2

    #creatinine 300-440 umol/l
    sofa_renal_creatinine.loc[(
        ((sofa_renal_creatinine['max_creatinine'] >= 300) &
        (sofa_renal_creatinine['max_creatinine'] <= 440))
    ), 'sofa_renal_score'] = 3

    #creatinine >440 umol/l
    sofa_renal_creatinine.loc[(
        ((sofa_renal_creatinine['max_creatinine'] > 440))
    ), 'sofa_renal_score'] = 4

    return sofa_renal_creatinine


def standardize_data(df_data_wide: pd.DataFrame, ignore_columns: list):
    """Function to make z-scores out of the numeric columns. Any columns in the ignore columns
    will not be touched.

    Args:
        df_data_wide_pd (_type_): DataFrame in wide format with the data
        ignore_columns (list): Columns not transformed

    Returns:
        _type_: _description_
    """
    numeric_features = df_data_wide.select_dtypes(include=['number']).columns.tolist()
    for ignored_col in ignore_columns:
        if ignored_col in numeric_features:
            numeric_features.remove(ignored_col)
        else:
            print(f"Warning, could not ignore column {ignored_col} since it was not in the numeric colunms!")
    df_data_wide[numeric_features] = df_data_wide[numeric_features].apply(lambda x: (x - x.mean()) / x.std(), axis=0)
    return df_data_wide


def get_windowed_data(df_data, time_window: str='12h'):
    def process_group(group_data):
        group_data = group_data.set_index("measurement_datetime")
        df_resampled = group_data.resample(time_window).mean() #CHANGE TIME WINDOW

        return df_resampled

    df_data_grouped = df_data.groupby(["visit_occurrence_id"])
    df_data_windowed = df_data_grouped.progress_apply(process_group)

    df_data_windowed = df_data_windowed.drop(columns=["visit_occurrence_id"]).reset_index()
    print(df_data_windowed.columns)
    df_data_windowed["seq_id"] = df_data_windowed.groupby(["visit_occurrence_id"]).cumcount()

    df_data_windowed = df_data_windowed.reset_index()
    df_data_windowed = df_data_windowed.set_index(["visit_occurrence_id", "measurement_datetime"])
    return df_data_windowed  



def transform_data(df_data_wide: pd.DataFrame, ignore_columns: list):
    """Performs a box-cox transformation of the data.

    Args:
        df_data_wide (pd.DataFrame): input data frame.
    """
    numeric_column = df_data_wide.select_dtypes(include=['number']).columns.tolist()
    for ignored_col in ignore_columns:
        if ignored_col in numeric_column:
            numeric_column.remove(ignored_col)
        else:
            print(f"Warning, could not ignore column {ignored_col} since it was not in the numeric colunms!")
    # plot for appendix
    n_cols = 5  # Number of columns in the grid
    n_rows = (len(numeric_column) // n_cols) + (len(numeric_column) % n_cols > 0)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3 * n_rows))
    axes = axes.flatten()
    # Iterate over features and apply Box-Cox transformation
    for i, feature in enumerate(numeric_column):
        valid_values = df_data_wide[feature].dropna()  # Exclude NaNs for Box-Cox
        transformed, _ = yeojohnson(
            valid_values
        )
        print("transforming", feature)
        
        if abs(skew(df_data_wide[feature].dropna().tolist())) > abs(skew(transformed)):
            df_data_wide.loc[valid_values.index, feature] = transformed
        
        sns.histplot((valid_values - valid_values.mean()) / valid_values.std(), kde=True, ax=axes[i], bins=15, color='red', alpha=0.7)  # Transformed
        sns.histplot((transformed - transformed.mean()) / transformed.std(), kde=True, ax=axes[i], bins=15, color='blue', alpha=0.4)  # Original
        axes[i].set_title(f"{feature}\nSK before: {skew(valid_values):.4f}\n SK after: {skew(transformed):.4f}")
        
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')

        plt.tight_layout()
        plt.savefig("./figures/transformations.png")

    return df_data_wide


def get_wide_measurements(df_measurements_long):
    ## Ensure all columns are available in the data
    # specifically drop concept_id and concept_name
    columns_to_keep = ["visit_occurrence_id", "measurement_datetime", "feature_name", "value_as_number"]
    df_measurements_long = df_measurements_long[columns_to_keep]

    # Sort the data by "measurement_datetime"
    df_measurements_long = df_measurements_long.sort_values(by="measurement_datetime")

    # Group the data to average duplicate measurements (preserve non-grouping columns)
    df_measurements_long = df_measurements_long.groupby(
        ["visit_occurrence_id", "measurement_datetime", "feature_name"],
        as_index=False
    ).mean(numeric_only=True)

    # Pivot the data to wide format
    df_measurements_wide = df_measurements_long.pivot(
        index=["visit_occurrence_id", "measurement_datetime"],
        columns="feature_name",
        values="value_as_number"
    )

    # Reset the index to make it a flat DataFrame
    return df_measurements_wide.reset_index()


def get_demographic_features(df_demo_wide: pd.DataFrame) -> pd.DataFrame:
    """Transforms the gender_source_value column to a binary gender column.

    Args:
        df_demo_wide (pd.DataFrame): Demographics DataFrame (wide).

    Returns:
        pd.DataFrame: Transformed df.
    """
    # Make gender binary
    df_demo_wide["Gender"] = np.nan
    df_demo_wide.loc[
        df_demo_wide['gender_source_value'].str.contains('Man', case=False, na=False),
        "Gender"
    ] = 1
    df_demo_wide.loc[
        df_demo_wide['gender_source_value'].str.contains('Vrouw', case=False, na=False),
        'Gender'
    ] = 0

    df_demo_wide["age_at_visit"] = df_demo_wide["age_at_visit"].astype(float)

    return df_demo_wide




def process_data(PROJECT_ID: str, config_gbq: dict, bq_client: Client=None, default_path: str="."):
    """Function to generate the dataframe for the states. Retrieves all relevant data from the database.

    Args:
        PROJECT_ID (str): Project id (google cloud).
        config_gbq (dict): Configuration config for the queries.
        bq_client (Client, optional): Client. Only has to be given when running as script (not in jupyter). Defaults to None.
        default_path (str, optional): Default path to the project. Defaults to ".".

    Returns:
        _type_: _description_
    """
    if not bq_client is None:
        credentials = bq_client._credentials
    else:
        credentials = None
    ### ------------------------------- SETUP Done ------------------------------- ###
    ### -------------------------------------------------------------------------- ###
    df_admissions = pd_gbq.read_gbq(combined_diagnoses_query, project_id=PROJECT_ID, configuration=config_gbq, use_bqstorage_api=True, credentials=credentials)
    df_combined_diagnoses = pd_gbq.read_gbq(combined_diagnoses_query, project_id=PROJECT_ID, configuration=config_gbq, use_bqstorage_api=True, credentials=credentials)
    
    
    df_sofa_resp = pd_gbq.read_gbq(query_sofa_respiratory, project_id=PROJECT_ID, configuration=config_gbq, use_bqstorage_api=True, credentials=credentials)
    df_sofa_resp = get_sofa_resp(df_sofa_resp)
    
    
    df_sofa_coagulation = pd_gbq.read_gbq(query_sofa_coagulation, project_id=PROJECT_ID, configuration=config_gbq, use_bqstorage_api=True, credentials=credentials)
    df_sofa_coagulation = get_sofa_coagulation(df_sofa_coagulation)


    df_sofa_liver = pd_gbq.read_gbq(query_sofa_liver, project_id=PROJECT_ID, configuration=config_gbq, use_bqstorage_api=True, credentials=credentials)
    df_sofa_liver = get_sofa_liver(df_sofa_liver)

    df_sofa_cardiovascular_meds = pd_gbq.read_gbq(query_vasopressors_ionotropes, project_id=PROJECT_ID, configuration=config_gbq, use_bqstorage_api=True, credentials=credentials)
    df_sofa_cardiovascular_meds = get_cardiovascular_meds(df_sofa_cardiovascular_meds)

    df_blood_pressure = pd_gbq.read_gbq(query_blood_pressure, project_id=PROJECT_ID, configuration=config_gbq, use_bqstorage_api=True, credentials=credentials)
    df_blood_pressure = get_blood_pressure(df_blood_pressure)

    #combine the scores from MAP and cardiovascular medication
    df_sofa_cardiovascular = pd.concat([df_blood_pressure, df_sofa_cardiovascular_meds], sort=False).sort_values(by='visit_occurrence_id')


    df_sofa_cns = pd_gbq.read_gbq(query_sofa_cns, project_id=PROJECT_ID, configuration=config_gbq, use_bqstorage_api=True, credentials=credentials)
    df_sofa_cns = get_sofa_cns(df_sofa_cns)


    df_sofa_renal = pd_gbq.read_gbq(query_sofa_renal, project_id=PROJECT_ID, configuration=config_gbq, use_bqstorage_api=True, credentials=credentials)
    df_sofa_renal = get_sofa_renal(df_sofa_renal)


    df_sofa_creatinine = pd_gbq.read_gbq(query_sofa_creatinine, project_id=PROJECT_ID, configuration=config_gbq, use_bqstorage_api=True, credentials=credentials)
    df_sofa_creatinine = get_sofa_creatinine(df_sofa_creatinine)

    df_sofa_renal = pd.concat([df_sofa_creatinine, df_sofa_renal], sort=False).sort_values(by='visit_occurrence_id')

    df_septic_shock = pd_gbq.read_gbq(query_lactate_sepsis, project_id=PROJECT_ID, configuration=config_gbq, use_bqstorage_api=True, credentials=credentials)
    df_septic_shock = get_initial_septic_shock(df_septic_shock)
    ### --- TOTAL SOFA SCORE ---- ###
    #merge the scores
    sofa = df_admissions.reset_index()['visit_occurrence_id']

    for df_to_merge, col_name in [
        (df_sofa_resp, "sofa_respiration_score"),
        (df_sofa_resp, "pf_ratio"),
        (df_sofa_resp, "fio2"),
        (df_sofa_resp, "pao2"),
        (df_sofa_coagulation, "sofa_coagulation_score"),
        (df_sofa_liver, "sofa_liver_score"),
        (df_sofa_cardiovascular, "sofa_cardiovascular_score"),
        (df_sofa_cns, "sofa_cns_score"),
        (df_sofa_renal, "sofa_renal_score"),
        (df_septic_shock, "septic_shock"),
    ]:
        to_merge = df_to_merge\
            .groupby('visit_occurrence_id')[col_name]\
            .max()\
            .to_frame(col_name)\
            .sort_values(by=['visit_occurrence_id'])\
            .reset_index()
        sofa = pd.merge(
            sofa,
            to_merge,
            on='visit_occurrence_id',
            how='left'
        )

    #max respiration score
    total_scores = sofa.set_index('visit_occurrence_id').sum(axis=1, skipna=True).to_frame('sofa_total_score')
    df_sofa = pd.merge(sofa, total_scores, on='visit_occurrence_id', how='left')

    df_combined_diagnoses = pd.merge(df_combined_diagnoses, df_sofa, on='visit_occurrence_id', how='inner')
    df_combined_diagnoses.loc[df_combined_diagnoses["septic_shock"].isna(), "septic_shock"] = False
    
    df_sepsis_at_admission = df_combined_diagnoses[(
        (df_combined_diagnoses['sepsis_at_admission'] == 1) |
        (
            (df_combined_diagnoses['sepsis_antibiotics_bool'] == 1) &
            (df_combined_diagnoses['sepsis_cultures_bool'] == 1) &
            (df_combined_diagnoses['sofa_total_score'] >= 2)
        )
    )]
    print("INFO: ")
    print(f"Unique septic persons: {len(df_sepsis_at_admission['person_id'].unique())}")
    print(f"Unique septic admissions: {len(df_sepsis_at_admission['visit_occurrence_id'].unique())}")
    # This will be septic shock!!!
    df_sepsis_at_admission = df_sepsis_at_admission[df_sepsis_at_admission["septic_shock"]==True]

    print("INFO: ")
    print(f"Unique septic shock persons: {len(df_sepsis_at_admission['person_id'].unique())}")
    print(f"Unique septic shock admissions: {len(df_sepsis_at_admission['visit_occurrence_id'].unique())}")

    sepsis_cases = df_sepsis_at_admission[["visit_occurrence_id", "person_id"]]
    sepsis_persons = tuple(sepsis_cases["person_id"].tolist())
    septic_shock_visit_ids = tuple(sepsis_cases["visit_occurrence_id"].tolist())
    print(f"Septic shock patients: {len(sepsis_cases)}")

    ### ------------------------- Training Data Processing ------------------------- ###
    sepsis_features = pd.read_csv(f"{default_path}/data/sepsis_features.csv")
    
    # remove unused features
    sepsis_features = sepsis_features[sepsis_features["feature_name"] != "_"]
    
    concept_ids = tuple(
        sepsis_features["concept_id"]\
            [sepsis_features["concept_id"].notna()]\
            .tolist()
    )

    weight_temp_concept_ids = tuple(
        sepsis_features[
            (sepsis_features["feature_name"] == "Temperature") |\
            (sepsis_features["feature_name"] == "Weight")
        ]["concept_id"].tolist()
    )


    df_demographics_wide = pd_gbq.read_gbq(
        query_demographics.format(visit_occurrence_ids=septic_shock_visit_ids),
        project_id=PROJECT_ID,
        configuration=config_gbq,
        use_bqstorage_api=True,
        credentials=credentials
    )

    df_demographics_wide = get_demographic_features(df_demographics_wide)

    print(f"Unique features being loaded {sepsis_features['feature_name'][sepsis_features['concept_id'].isin(concept_ids)].unique()}")
    mq = query_measurement.format(visit_occurrence_ids=septic_shock_visit_ids, weight_temp_ids=weight_temp_concept_ids, concept_ids=concept_ids)

    df_measurements_long = pd_gbq.read_gbq(  # This includes all lab values and the body weight
        mq,
        project_id=PROJECT_ID,
        configuration=config_gbq,
        use_bqstorage_api=True,
        credentials=credentials
    )
    print(f"Cols : {df_measurements_long.columns}")

    #df_measurements_long["value_as_number"] = df_measurements_long["value_as_number"].astype(float)
    df_measurements_long["measurement_datetime"] = pd.to_datetime(df_measurements_long["measurement_datetime"])
    df_measurements_long["value_as_number"] = pd.to_numeric(df_measurements_long["value_as_number"], errors="coerce")
    # Add feature names
    df_measurements_long = pd.merge(
        df_measurements_long,
        sepsis_features[["feature_name","concept_id"]],
        left_on='measurement_concept_id',
        right_on="concept_id",
        how='left'
    )

    print(f"Measurements for temp: {len(df_measurements_long[df_measurements_long['feature_name'] == 'Temperature'])}")

    n_before = df_measurements_long.count().sum()
    print(f"Datapoints before: {df_measurements_long.count().sum()}")
    df_measurements_long = drop_outliers_iqr_long(df_measurements_long, df_measurements_long["feature_name"].unique())
    print(f"Dropped {n_before-df_measurements_long.count().sum()} outliers.")
    print(f"{df_measurements_long.count().sum()} values are still present.")

    index_long = df_measurements_long.head(10).index
    df_measurements_wide = get_wide_measurements(df_measurements_long)
    print(f"Index before: {index_long}, after: {df_measurements_wide.head(10).index}")
    print(f"Wide temp dps: {df_measurements_wide['Temperature'].notna().sum()}")
    print(F"Wide temp rows: {len(df_measurements_wide)}")
    print(df_measurements_wide.head(5))
    print(df_measurements_wide.columns)

    # df_demographics_wide = pd.merge(
    #     df_sepsis_at_admission,
    #     df_demographics_wide
    #     how="left",
    #     on="person_id"
    # )[["visit_occurrent_id", "gender_source_value", "year_of_birth"]]
    # df_sepsis_data_wide = pd.merge(
    #     df_measurements_wide,
    #     df_demographics_wide,
    #     how="left",
    #     on="visit_occurrent_id")

    ### -------------------- Merge Additional Columns Here ------------------------ ###
    # 1. Note that outliers should be dropped!
    
    ## Add sofa cns score
    df_sepsis = df_measurements_wide.merge(
        df_sepsis_at_admission[["visit_occurrence_id", "sofa_cns_score"]],
        on=["visit_occurrence_id"],
        how="left"
    )

    df_sepsis["sofa_cns_score"] = df_sepsis["sofa_cns_score"].astype(float)
    #df_sepsis['gender_source_value'] = pd.to_numeric(df_sepsis['gender_source_value'], errors='coerce')

    # Add pao2/fio2
    df_ventilation_wide = pd_gbq.read_gbq(
        query_ventilation.format(visit_occurrence_ids=septic_shock_visit_ids),
        project_id=PROJECT_ID,
        configuration=config_gbq,
        use_bqstorage_api=True,
        credentials=credentials
    )
    
    df_ventilation_wide["pao2_fio2_ratio"] = df_ventilation_wide["pao2"].astype(float) / df_ventilation_wide["fio2"].astype(float)
    
    df_sepsis = df_sepsis.merge(
        df_ventilation_wide[
            [
                "visit_occurrence_id",
                "measurement_datetime",
                "ventilatory_support",
                "pao2_fio2_ratio",
                "pao2"
            ]
        ],
        how='outer',
        on=["visit_occurrence_id", "measurement_datetime"]
    )

    # drop unreasonable values
    #df_sepsis.loc[df_sepsis['pao2'] >= 400, 'pao2'] = np.nan

    df_sepsis = drop_outliers_iqr_wide(
        df_sepsis,
        ["pao2_fio2_ratio", "pao2"]
    )

    #df_sepsis["ventilatory_support"] = df_sepsis["ventilatory_support"].astype(bool)
    df_sepsis["pao2_fio2_ratio"] = df_sepsis["pao2_fio2_ratio"].astype(float)
    df_sepsis["pao2"] = df_sepsis["pao2"].astype(float)

    df_sepsis = df_sepsis.dropna(how="all")  # drop empty rows!

    df_sepsis = pd.merge(
        df_sepsis,
        df_demographics_wide[["visit_occurrence_id", "Gender", "age_at_visit"]],
        on="visit_occurrence_id",
        how="left"
    )


    # ----------------------  Aggregate values to 4h values  ---------------------- #
    df_sepsis_windows = get_windowed_data(df_sepsis)
    result_non_na_cells = df_sepsis_windows.count().sum()
    print(f"dataset has {result_non_na_cells} non-na entries!")
    
    trans_ignore_cols = ["index", "Gender", "ventilatory_support", "seq_id"]
    # Transforming data with box-cox
    df_more_normal = transform_data(df_sepsis_windows, ignore_columns=trans_ignore_cols)
    
    # Standardizing data for imputation and clustering
    df_standardized = standardize_data(df_more_normal, ignore_columns=trans_ignore_cols)

    df_standardized = df_standardized.drop(columns=["index", "seq_id"])

    # plot some stuff
    print(df_standardized.describe())
    plot_missing_data(df_standardized, imputation_type="No imputation").savefig(f"{default_path}/figures/initial_missing_data.png")
    # Apply the function to each column in the DataFrame (excluding metadata columns)
    df_standardized = df_standardized.sort_index()
    df_average_nans = pd.DataFrame(
        df_standardized.iloc[:, 6:].apply(calculate_average_nans, axis=0),  # Adjust slice as needed for features
        columns=["average_nans"]
    )
    # Convert the result into a DataFrame for clarity
    df_average_nans.to_csv(f"{default_path}/figures/feature_gaps_size.csv")
    
    df_standardized = drop_na_cols(df_standardized, threshold=0.3)
    
    # imputation
    average_nans_per_column_filtered = df_standardized.iloc[:, :].apply(calculate_average_nans, axis=0)

    # Perform forward fill with the calculated averages
    # Pass the correct variable 'average_nans_per_column_filtered'
    df_standardized_filled = forward_fill_by_column(df_standardized, average_nans_per_column_filtered)
    plot_missing_data(df_standardized_filled, imputation_type="FF imputation").savefig(f"{default_path}/figures/ff_missing_data.png")

        
    # Get unique visit occurrences
    unique_visits = df_standardized_filled.index.get_level_values("visit_occurrence_id").unique()

    # Split unique visit occurrences into train, validation, and test
    train_visits, temp_visits = train_test_split(unique_visits, test_size=0.4, random_state=42)  # 60% train
    valid_visits, test_visits = train_test_split(temp_visits, test_size=0.5, random_state=42)  # 20% validation, 20% test

    # Create train, validation, and test sets based on visit occurrences
    train_data = df_standardized_filled[df_standardized_filled.index.get_level_values("visit_occurrence_id").isin(train_visits)]
    valid_data = df_standardized_filled[df_standardized_filled.index.get_level_values("visit_occurrence_id").isin(valid_visits)]
    test_data = df_standardized_filled[df_standardized_filled.index.get_level_values("visit_occurrence_id").isin(test_visits)]

    print("Train data shape:", train_data.shape)
    print("Validation data shape:", valid_data.shape)
    print("Test data shape:", test_data.shape)

    # Define a function to apply KNN imputation for each visit occurrence
    # Apply the KNN imputation with dynamic k
    print("Starting k means imputation")
    knn_imputer = KNNImputer(n_neighbors=5)
    # Apply KNN imputation to the entire dataset
    imputed_values_train = knn_imputer.fit_transform(train_data)

    # Create a new DataFrame with the imputed values
    df_knn_imputed_train = pd.DataFrame(
        imputed_values_train,
        columns=train_data.columns,
        index=train_data.index
    )#knn_impute_by_column(train_data)
    print('train data done')
    df_knn_imputed_val = pd.DataFrame(
        knn_imputer.transform(valid_data),
        columns=valid_data.columns,
        index=valid_data.index
    )#knn_impute_by_column(valid_data)
    print('validation data done')
    df_knn_imputed_test = pd.DataFrame(
        knn_imputer.transform(test_data),
        columns=test_data.columns,
        index=test_data.index
    )#knn_impute_by_column(test_data)


    print('test data done')
    print("Imputation done!")

    plot_missing_data(df_knn_imputed_train, 'Train KNN').savefig(f"{default_path}/figures/ffknn_train_missing_data.png")
    plot_missing_data(df_knn_imputed_val, 'Validation KNN').savefig(f"{default_path}/figures/ffknn_val_missing_data.png")
    plot_missing_data(df_knn_imputed_test, 'Test KNN').savefig(f"{default_path}/figures/ffknn_test_missing_data.png")

    # Apply the function to each dataset
    df_knn_imputed_train, dropped_features_train = drop_features_with_missing_data(df_knn_imputed_train, threshold=40)
    df_knn_imputed_val, dropped_features_val = drop_features_with_missing_data(df_knn_imputed_val, threshold=40)
    df_knn_imputed_test, dropped_features_test = drop_features_with_missing_data(df_knn_imputed_test, threshold=40)

    # Print the columns dropped in each dataset
    print("Columns dropped from training data:", dropped_features_train)
    print("Columns dropped from validation data:", dropped_features_val)
    print("Columns dropped from test data:", dropped_features_test)

    # convert binary to binary after kNN
    for col in ["Gender", "ventilatory_support"]:
        df_knn_imputed_train[col] = df_knn_imputed_train[col].round().astype(int)
        df_knn_imputed_val[col] = df_knn_imputed_val[col].round().astype(int)
        df_knn_imputed_test[col] = df_knn_imputed_test[col].round().astype(int)

    # Perform linear interpolation on remaining missing values
    #df_filled_knn_interp_train = df_knn_imputed_train.interpolate(method='linear', axis=0, limit_direction='both')
    #df_filled_knn_interp_val = df_knn_imputed_val.interpolate(method='linear', axis=0, limit_direction='both')
    #df_filled_knn_interp_test = df_knn_imputed_test.interpolate(method='linear', axis=0, limit_direction='both')
    return df_knn_imputed_train, df_knn_imputed_val, df_knn_imputed_test


if __name__ == '__main__':
    ### ------------------- Project setup with Google Cloud + Bigquery ------------------- ###
    # Follow tutorial to setup: https://cloud.google.com/iam/docs/keys-create-delete#python
    
    PROJECT_ID = "mrih-440308" # REPLACE IF RUN LOCALLY!!
    DATASET_PROJECT_ID = 'amsterdamumcdb'
    DATASET_ID = 'version1_5_0'
    LOCATION = 'eu'

    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'C:\\Users\\MR\\AppData\\Roaming\\gcloud\\application_default_credentials.json'
    bq_client = Client(project=PROJECT_ID, location=LOCATION)
    print('Authenticated! :)')

    config_gbq = {
        'query': {
            'defaultDataset': {
                "datasetId": DATASET_ID,
                "projectId": DATASET_PROJECT_ID
            },
            'Location': LOCATION
        }
    }

    df_filled_knn_interp_train, df_filled_knn_interp_val, df_filled_knn_interp_test = process_data(PROJECT_ID, config_gbq, bq_client)

    # Step 1: Compute Correlation Matrix
    correlation_matrix = df_filled_knn_interp_train.corr()

    # Step 2: Plot Heatmap
    plt.figure(figsize=(12, 10))  # Set the size of the heatmap
    sns.heatmap(
        correlation_matrix,
        annot=True,        # Annotate cells with correlation values
        fmt=".2f",         # Format for decimal places
        cmap="coolwarm",   # Color map
        cbar=True,         # Show color bar
        annot_kws={"size": 4}

    )
    plt.title("Correlation Matrix Heatmap")
    plt.savefig(f"./figures/correlation_matrix.png", dpi=400, bbox_inches="tight")

    df_filled_knn_interp_train = df_filled_knn_interp_train.drop(columns=["Bicarbonate"])
    df_filled_knn_interp_val = df_filled_knn_interp_val.drop(columns=["Bicarbonate"])
    df_filled_knn_interp_test = df_filled_knn_interp_test.drop(columns=["Bicarbonate"])
    # Display the resulting DataFrame
    print(f"Remaining train missing: {df_filled_knn_interp_train.isna().sum()}")
    print(f"Remaining val missing: {df_filled_knn_interp_val.isna().sum()}")
    print(f"Remaining test missing: {df_filled_knn_interp_test.isna().sum()}")
    
    # Clustering
    #kmeans, cluster_centers = cluster_kmpp(df_filled_knn_interp_train, k=200)

    #print(cluster_centers)
    #pd.DataFrame({"state": cluster_centers}, index=df_filled_knn_interp_train.index).to_csv(f"./data/train_centers.csv")
    #pd.DataFrame({"state": kmeans.predict(df_filled_knn_interp_val.values)}, index=df_filled_knn_interp_val.index).to_csv(f"./data/val_centers.csv")
    #pd.DataFrame({"state": kmeans.predict(df_filled_knn_interp_test.values)}, index=df_filled_knn_interp_test.index).to_csv(f"./data/test_centers.csv")

    #res, fig = evaluate_clustering(df_filled_knn_interp_train)
    #fig.savefig(f"./figures/clustering_eval.png")
    print(df_filled_knn_interp_train.dtypes)
    print(df_filled_knn_interp_train.describe())
    import optuna

    def optuna_objective(trial):
        # Define hyperparameters to tune
        latent_dim = trial.suggest_categorical("latent_dim", [8, 16, 32])  # Latent space size
        layer_sizes = trial.suggest_categorical("layer_sizes", [tuple([64, 32, 16]), tuple([32, 16, 10]), tuple([128, 64, 32])])  # Layer architectures
        lr = trial.suggest_loguniform("lr", 1e-4, 1e-2)  # Learning rate
        batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])  # Batch sizes

        input_dim = len(df_filled_knn_interp_train.columns)# - 1  # Adjust based on your feature vector size
        max_epochs = 100

        print(f"Creating Autoencoder with {input_dim} input size and layers {layer_sizes}")
        autoencoder = Autoencoder(input_dim, latent_dim=latent_dim, layer_sizes=list(layer_sizes), k=400)
        criterion = DKNLoss()
        optimizer = optim.Adam(autoencoder.parameters(), lr=lr)

        # Dataset and DataLoader
        used_features = df_filled_knn_interp_train.columns
        train_icu_dataset = AmsICUSepticShock(df_filled_knn_interp_train[used_features])
        train_data_loader = DataLoader(train_icu_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

        
        val_icu_dataset = AmsICUSepticShock(df_filled_knn_interp_val[used_features])
        val_data_loader = DataLoader(val_icu_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

        print(f"Train data size: {len(train_icu_dataset)}")
        print(f"Val dataset size: {len(val_icu_dataset)}")

        # Move model to device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using device {device}")
        autoencoder = autoencoder.to(device)

        # Training
        metrics = training(
            autoencoder,
            optimizer,
            criterion,
            train_data_loader=train_data_loader,
            val_data_loader=val_data_loader,
            max_epochs=max_epochs,
            model_path=f"./models/dkm_model_{trial.number}.pth",
            verbose=False
        )
        # Sample data (replace with actual data)
        train_losses = metrics["train_loss"]  # Example loss values over epochs
        val_losses = metrics["val_loss"]

        # Plot training loss
        plt.figure(figsize=(10, 6))
        # total loss
        sns.lineplot(x=range(len(train_losses)), y=train_losses, label="Training Loss", linewidth=2)
        sns.lineplot(x=range(len(val_losses)), y=val_losses, label="Validation Loss", linewidth=2)

        # reconstruction loss
        sns.lineplot(x=range(len(metrics["train_loss_rec"])), y=metrics["train_loss_rec"], label="Training Loss Rec", linewidth=2)
        sns.lineplot(x=range(len(metrics["val_loss_rec"])), y=metrics["val_loss_rec"], label="Validation Loss Rec", linewidth=2)

        # cluster loss
        sns.lineplot(x=range(len(metrics["train_loss_clust"])), y=metrics["train_loss_clust"], label="Training Loss Clust", linewidth=2)
        sns.lineplot(x=range(len(metrics["val_loss_clust"])), y=metrics["val_loss_clust"], label="Validation Loss Clust", linewidth=2)

        plt.title("Training Loss Over Epochs", fontsize=16, fontweight="bold")
        plt.xlabel("Epoch", fontsize=14)
        plt.ylabel("Loss", fontsize=14)
        plt.grid(alpha=0.5)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig(f"./figures/dkm_model_{trial.number}.png")

        plt.figure(figsize=(10, 6))

        json.dump(metrics, open(f"./data/dkm_model_{trial.number}_metrics.json", 'w', encoding="utf-8"), indent=4)
        

        # Return the final training loss as the objective value (validation score!!)
        return metrics["val_loss"][-1]
    
    storage = "sqlite:///optuna_study_final.db"
    study = optuna.create_study(
        storage=storage,
        study_name="deep_kmeans_autoencoder",
        direction="minimize",  # Minimize training loss
        load_if_exists=True
    )

    # Optimize the study
    study.optimize(optuna_objective, n_trials=50)

    # Print the best hyperparameters
    print("Best hyperparameters:")
    print(study.best_params)

    # Save the study results
    study.trials_dataframe().to_csv("optuna_study_results.csv")
    # Instantiate and train the autoencoder
    # used_features = df_filled_knn_interp_train.columns
    # used_features.remove("index")

    # #print(df_sepsis_time_windows.columns)
    # input_dim = len(used_features)  # Adjust based on your feature vector size
    # epochs = 30
    # batch_size = 128
    # latent_dim = 20

    # print(f"Creating Autoencoder with {len(used_features)} input size")
    # autoencoder = Autoencoder(input_dim, latent_dim=latent_dim, layer_sizes=[32, 16, 10], k=400)
    # #print(autoencoder)
    # criterion = DKNLoss()
    # optimizer = optim.Adam(autoencoder.parameters(), lr=1e-3)

    # icu_dataset = AmsICUSepticShock(df_filled_knn_interp_train[used_features])
    # data_loader = DataLoader(icu_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    # # Tracked data/metrics

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print(f"Using device {device}")
    # autoencoder = autoencoder.to(device)

    # metrics = training(
    #     autoencoder,
    #     optimizer,
    #     criterion,
    #     data_loader,
    #     epochs=epochs,
    #     model_path='./models/test_model.pth',
    #     verbose=True
    # )

