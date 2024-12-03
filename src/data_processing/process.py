"""Main module to process the Data from the AmsterdamUMCDB.

Author: Mika Florin Rosin
"""
import os

import pandas as pd
import pandas_gbq as pd_gbq
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import boxcox, skew
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from google.cloud.bigquery.client import Client
from google.cloud import bigquery

from src.data_processing.queries.combined_diagnosis import combined_diagnoses_query
from src.data_processing.queries.sofa_scores import *
from src.data_processing.queries.data_queries import query_demographics, query_measurement
from src.data_processing.missing_data import plot_missing_data, calculate_average_nans, drop_na_cols
from src.data_processing.data_imputation import calculate_average_nans, knn_impute_by_column, forward_fill_by_column, forward_fill_limited, drop_features_with_missing_data
from src.clustering import cluster_kmpp, evaluate_clustering

tqdm.pandas()


def get_sofa_resp(df_sofa_resp):
    #remove extreme outliers
    df_sofa_resp.loc[(df_sofa_resp['fio2'] > 100), 'fio2'] = np.nan
    #convert FiO2 in % to fraction
    df_sofa_resp.loc[(df_sofa_resp['fio2'] <= 100) & (df_sofa_resp['fio2'] >= 20) , 'fio2'] = df_sofa_resp['fio2']/100
    fio2_cleaned = df_sofa_resp['fio2'].dropna()
    pao2_cleaned = df_sofa_resp['pao2'].dropna().astype(float)
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

    return df_sofa_resp, fio2_cleaned, pao2_cleaned


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


def drop_outliers(df_data):
    """Removed outliers from all features, that are outside of the
    1.5 * IQR range.
    Args:
        df_data (_type_): dataframe with the data.

    Returns:
        _type_: _description_
    """
    columns = df_data['feature_name'].unique()
    for feature in tqdm(columns, desc=f"processing f{len(columns)} features"):
        # Select data for the current concept
        feature_values = np.array(df_data[df_data['feature_name'] == feature]['value_as_number']\
            .dropna()\
            .astype(float)\
            .tolist()
        )
        Q1 = np.percentile(feature_values, 25)  # 25th percentile
        Q3 = np.percentile(feature_values, 75)  # 75th percentile
        IQR = Q3 - Q1

        # Define bounds
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Filter data to remove outliers
        df_data = df_data[
            (df_data['feature_name'] != feature) |
            (
                (df_data["value_as_number"] >= lower_bound) &
                (df_data["value_as_number"] <= upper_bound)
            )
        ]
    return df_data

def standardize_data(df_data_wide):
    numeric_features = df_data_wide.select_dtypes(include=['number']).columns.tolist()
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



def transform_data(df_data_wide: pd.DataFrame):
    """Performs a box-cox transformation of the data.

    Args:
        df_data_wide (pd.DataFrame): input data frame.
    """
    numeric_column = df_data_wide.select_dtypes(include=['number']).columns.tolist()
    # Iterate over features and apply Box-Cox transformation
    for feature in numeric_column:
        valid_values = df_data_wide[feature].dropna()  # Exclude NaNs for Box-Cox
        transformed, _ = boxcox(
            valid_values + ((abs(valid_values.min()) + 1) if valid_values.min() <= 0 else 0)
        )
        print("transforming", feature)
        if abs(skew(df_data_wide[feature].dropna().tolist())) > abs(skew(transformed)):
            
            df_data_wide.loc[valid_values.index, feature] = transformed
    return df_data_wide


def get_wide_measurements(df_measurements_long):
    ## Ensure all columns are available in the data
    # specifically drop concept_id and concept_name
    columns_to_keep = ["visit_occurrence_id", "measurement_datetime", "feature_name", "value_as_number", "year_of_birth", "gender_source_value"]
    df_measurements_long = df_measurements_long[columns_to_keep]

    # Sort the data by "measurement_datetime"
    df_measurements_long = df_measurements_long.sort_values(by="measurement_datetime")

    # Group the data to average duplicate measurements (preserve non-grouping columns)
    df_measurements_long = df_measurements_long.groupby(
        ["visit_occurrence_id", "measurement_datetime", "feature_name", "gender_source_value", "year_of_birth"],
        as_index=False
    ).mean(numeric_only=True)

    # Pivot the data to wide format
    df_measurements_wide = df_measurements_long.pivot(
        index=["visit_occurrence_id", "measurement_datetime", "gender_source_value", "year_of_birth"],
        columns="feature_name",
        values="value_as_number"
    )

    # Reset the index to make it a flat DataFrame
    return df_measurements_wide.reset_index()


def process_data(PROJECT_ID, config_gbq, local=False):
    ### ------------------------------- SETUP Done ------------------------------- ###
    ### -------------------------------------------------------------------------- ###
    df_admissions = pd_gbq.read_gbq(combined_diagnoses_query, project_id=PROJECT_ID, configuration=config_gbq, use_bqstorage_api=True, credentials=bq_client._credentials if local else None)
    df_combined_diagnoses = pd_gbq.read_gbq(combined_diagnoses_query, project_id=PROJECT_ID, configuration=config_gbq, use_bqstorage_api=True, credentials=bq_client._credentials if local else None)
    
    
    df_sofa_resp = pd_gbq.read_gbq(query_sofa_respiratory, project_id=PROJECT_ID, configuration=config_gbq, use_bqstorage_api=True, credentials=bq_client._credentials if local else None)
    df_sofa_resp, df_fio2_cleaned, df_pao2_cleaned = get_sofa_resp(df_sofa_resp)
    
    
    df_sofa_coagulation = pd_gbq.read_gbq(query_sofa_coagulation, project_id=PROJECT_ID, configuration=config_gbq, use_bqstorage_api=True, credentials=bq_client._credentials if local else None)
    df_sofa_coagulation = get_sofa_coagulation(df_sofa_coagulation)


    df_sofa_liver = pd_gbq.read_gbq(query_sofa_liver, project_id=PROJECT_ID, configuration=config_gbq, use_bqstorage_api=True, credentials=bq_client._credentials if local else None)
    df_sofa_liver = get_sofa_liver(df_sofa_liver)

    df_sofa_cardiovascular_meds = pd_gbq.read_gbq(query_vasopressors_ionotropes, project_id=PROJECT_ID, configuration=config_gbq, use_bqstorage_api=True, credentials=bq_client._credentials if local else None)
    df_sofa_cardiocascular_meds = get_cardiovascular_meds(df_sofa_cardiovascular_meds)

    df_blood_pressure = pd_gbq.read_gbq(query_blood_pressure, project_id=PROJECT_ID, configuration=config_gbq, use_bqstorage_api=True, credentials=bq_client._credentials if local else None)
    df_blood_pressure = get_blood_pressure(df_blood_pressure)

    #combine the scores from MAP and cardiovascular medication
    df_sofa_cardiovascular = pd.concat([df_blood_pressure, df_sofa_cardiovascular_meds], sort=False).sort_values(by='visit_occurrence_id')


    df_sofa_cns = pd_gbq.read_gbq(query_sofa_cns, project_id=PROJECT_ID, configuration=config_gbq, use_bqstorage_api=True, credentials=bq_client._credentials if local else None)
    df_sofa_cns = get_sofa_cns(df_sofa_cns)


    df_sofa_renal = pd_gbq.read_gbq(query_sofa_renal, project_id=PROJECT_ID, configuration=config_gbq, use_bqstorage_api=True, credentials=bq_client._credentials if local else None)
    df_sofa_renal = get_sofa_renal(df_sofa_renal)


    df_sofa_creatinine = pd_gbq.read_gbq(query_sofa_creatinine, project_id=PROJECT_ID, configuration=config_gbq, use_bqstorage_api=True, credentials=bq_client._credentials if local else None)
    df_sofa_creatinine = get_sofa_creatinine(df_sofa_creatinine)

    sofa_renal = pd.concat([df_sofa_creatinine, df_sofa_renal], sort=False).sort_values(by='visit_occurrence_id')
    ### --- TOTAL SOFA SCORE ---- ###
    #merge the scores
    sofa = df_admissions.reset_index()['visit_occurrence_id']

    for df_to_merge, col_name in [
        (df_sofa_resp, "sofa_respiration_score"),
        (df_sofa_coagulation, "sofa_coagulation_score"),
        (df_sofa_liver, "sofa_liver_score"),
        (df_sofa_cardiovascular, "sofa_cardiovascular_score"),
        (df_sofa_cns, "sofa_cns_score"),
        (df_sofa_renal, "sofa_renal_score"),
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
    df_sepsis_at_admission = df_combined_diagnoses[(
        (df_combined_diagnoses['sepsis_at_admission'] == 1) |
        (
            (df_combined_diagnoses['sepsis_antibiotics_bool'] == 1) &
            (df_combined_diagnoses['sepsis_cultures_bool'] == 1) &
            (df_combined_diagnoses['sofa_total_score'] >= 2)
        )
    )]

    print("INFO: ")
    print(f"Unique persons: {len(df_sepsis_at_admission['person_id'].unique())}")
    print(f"Unique admissions: {len(df_sepsis_at_admission['visit_occurrence_id'].unique())}")

    sepsis_cases = df_sepsis_at_admission[["visit_occurrence_id", "person_id"]]
    sepsis_persons = tuple(sepsis_cases["person_id"].tolist())
    ### ------------------------- Training Data Processing ------------------------- ###
    sepsis_features = pd.read_csv("./data/sepsis_features.csv")
    sepsis_features = sepsis_features[sepsis_features["feature_name"] != "_"]
    print(f"Unique features being loaded {sepsis_features['feature_name'].unique()}")
    concept_ids = tuple(sepsis_features["concept_id"].tolist())
    weight_temp_concept_ids = tuple(
        sepsis_features[
            (sepsis_features["feature_name"] == "Temperature") |\
            (sepsis_features["feature_name"] == "Weight")
        ]["concept_id"].tolist()
    )


    df_demographics_wide = pd_gbq.read_gbq(
        query_demographics.format(person_ids=sepsis_persons),
        project_id=PROJECT_ID,
        configuration=config_gbq,
        use_bqstorage_api=True,
        credentials=bq_client._credentials if local else None
    )
    df_demographics_wide.loc[
        df_demographics_wide['gender_source_value'].str.contains('Man', case=False, na=False),
        'gender_source_value'
    ] = 1
    df_demographics_wide.loc[
        df_demographics_wide['gender_source_value'].str.contains('Vrouw', case=False, na=False),
        'gender_source_value'
    ] = 0

    mq = query_measurement.format(person_ids=sepsis_persons, weight_temp_ids=weight_temp_concept_ids, concept_ids=concept_ids)
    print(mq)
    df_measurements_long = pd_gbq.read_gbq(  # This includes all lab values and the body weight
        mq,
        project_id=PROJECT_ID,
        configuration=config_gbq,
        use_bqstorage_api=True,
        credentials=bq_client._credentials if local else None
    )
    
    df_measurements_long = pd.merge(
        df_measurements_long,
        df_demographics_wide,
        on="person_id",
        how="left"
    )
    print(f"Cols with dem: {df_measurements_long.columns}")

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
    df_measurements_long = drop_outliers(df_measurements_long)
    print(f"Dropped {n_before-df_measurements_long.count().sum()} outliers.")
    print(f"{df_measurements_long.count().sum()} values are still present.")

    index_long = df_measurements_long.head(10).index
    df_measurements_wide = get_wide_measurements(df_measurements_long)
    print(f"Index before: {index_long}, after: {df_measurements_wide.head(10).index}")
    print(f"Wide temp dps: {df_measurements_wide['Temperature'].notna().sum()}")
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
    ## Add sofa cns score
    df_sepsis = df_measurements_wide.merge(
        df_sepsis_at_admission[["visit_occurrence_id", "sofa_cns_score"]],
        on=["visit_occurrence_id"],
        how="left"
    )

    df_sepsis['gender_source_value'] = pd.to_numeric(df_sepsis['gender_source_value'], errors='coerce')

    # Aggregate values to 4h values
    df_sepsis_windows = get_windowed_data(df_sepsis)
    result_non_na_cells = df_sepsis_windows.count().sum()
    print(f"dataset has {result_non_na_cells} non-na entries!")

    # Transforming data with box-cox
    df_more_normal = transform_data(df_sepsis_windows)
    
    # Standardizing data for imputation and clustering
    df_standardized = standardize_data(df_more_normal)

    # plot some stuff
    print(df_standardized.describe())
    plot_missing_data(df_standardized, imputation_type="No imputation").savefig("./figures/initial_missing_data.png")
    # Apply the function to each column in the DataFrame (excluding metadata columns)
    df_standardized = df_standardized.sort_index()
    df_average_nans = pd.DataFrame(
        df_standardized.iloc[:, 6:].apply(calculate_average_nans, axis=0),  # Adjust slice as needed for features
        columns=["average_nans"]
    )
    # Convert the result into a DataFrame for clarity
    df_average_nans.to_csv("./figures/feature_gaps_size.csv")
    
    df_standardized = drop_na_cols(df_standardized, threshold=0.3)
    
    # imputation
    average_nans_per_column_filtered = df_standardized.iloc[:, :].apply(calculate_average_nans, axis=0)

    # Perform forward fill with the calculated averages
    # Pass the correct variable 'average_nans_per_column_filtered'
    df_standardized_filled = forward_fill_by_column(df_standardized, average_nans_per_column_filtered)
    plot_missing_data(df_standardized_filled, imputation_type="FF imputation").savefig("./figures/ff_missing_data.png")

        
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
    df_knn_imputed_train = knn_impute_by_column(train_data)
    print('train data done')
    df_knn_imputed_val = knn_impute_by_column(valid_data)
    print('validation data done')
    df_knn_imputed_test = knn_impute_by_column(test_data)
    print('test data done')

    plot_missing_data(df_knn_imputed_train, 'Train KNN').savefig("./figures/ffknn_train_missing_data.png")
    plot_missing_data(df_knn_imputed_val, 'Validation KNN').savefig("./figures/ffknn_val_missing_data.png")
    plot_missing_data(df_knn_imputed_test, 'Test KNN').savefig("./figures/ffknn_test_missing_data.png")

    # Apply the function to each dataset
    df_knn_imputed_train, dropped_features_train = drop_features_with_missing_data(df_knn_imputed_train, threshold=40)
    df_knn_imputed_val, dropped_features_val = drop_features_with_missing_data(df_knn_imputed_val, threshold=40)
    df_knn_imputed_test, dropped_features_test = drop_features_with_missing_data(df_knn_imputed_test, threshold=40)

    # Print the columns dropped in each dataset
    print("Columns dropped from training data:", dropped_features_train)
    print("Columns dropped from validation data:", dropped_features_val)
    print("Columns dropped from test data:", dropped_features_test)

    # Perform linear interpolation on remaining missing values
    df_filled_knn_interp_train = df_knn_imputed_train.interpolate(method='linear', axis=0, limit_direction='both')
    df_filled_knn_interp_val = df_knn_imputed_val.interpolate(method='linear', axis=0, limit_direction='both')
    df_filled_knn_interp_test = df_knn_imputed_test.interpolate(method='linear', axis=0, limit_direction='both')
    return df_filled_knn_interp_train, df_filled_knn_interp_val, df_filled_knn_interp_test


if __name__ == '__main__':
    ### ------------------- Project setup with Google Cloud + Bigquery ------------------- ###
    # Follow tutorial to setup: https://cloud.google.com/iam/docs/keys-create-delete#python
    
    PROJECT_ID = "mrih-440308" # REPLACE IF RUN LOCALLY!!
    DATASET_PROJECT_ID = 'amsterdamumcdb'
    DATASET_ID = 'version1_5_0'
    LOCATION = 'eu'

    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'C:\\Users\\MR\\AppData\\Roaming\\gcloud\\application_default_credentials.json'
    bq_client = Client(location=LOCATION)
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

    df_filled_knn_interp_train, df_filled_knn_interp_val, df_filled_knn_interp_test = process_data(PROJECT_ID, config_gbq, True)

    # Display the resulting DataFrame
    print(f"Remaining train missing: {df_filled_knn_interp_train.isna().sum()}")
    print(f"Remaining train missing: {df_filled_knn_interp_val.isna().sum()}")
    print(f"Remaining train missing: {df_filled_knn_interp_test.isna().sum()}")
    
    # Clustering
    kmeans, cluster_centers = cluster_kmpp(df_filled_knn_interp_train, k=200)

    print(cluster_centers)
    pd.DataFrame({"state": cluster_centers}, index=df_filled_knn_interp_train.index).to_csv("./data/train_centers.csv")
    pd.DataFrame({"state": kmeans.predict(df_filled_knn_interp_val.values)}, index=df_filled_knn_interp_val.index).to_csv("./data/val_centers.csv")
    pd.DataFrame({"state": kmeans.predict(df_filled_knn_interp_test.values)}, index=df_filled_knn_interp_test.index).to_csv("./data/test_centers.csv")

    res, fig = evaluate_clustering(df_filled_knn_interp_train)
    fig.savfig("./figures/clustering_eval.png")
