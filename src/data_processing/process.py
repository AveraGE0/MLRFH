"""Main module to process the Data from the AmsterdamUMCDB.

Author: Mika Florin Rosin
"""
import os

import pandas as pd
import pandas_gbq as pd_gbq
import numpy as np

from google.cloud.bigquery.client import Client
from google.cloud import bigquery

from src.data_processing.queries.combined_diagnosis import combined_diagnosis_query
from src.data_processing.queries.sofa_scores import *


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
    df_sofa_resp =df_sofa_resp.dropna(subset=['pao2'])
    df_sofa_resp.loc[:,'pf_ratio'] =df_sofa_resp['pao2']/df_sofa_resp['fio2']

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


if __name__ == '__main__':
    ### ------------------- Project setup with Google Cloud + Bigquery ------------------- ###
    # Follow tutorial to setup: https://cloud.google.com/iam/docs/keys-create-delete#python
    
    PROJECT_ID = "mrih-440308" # REPLACE IF RUN LOCALLY!!
    DATASET_PROJECT_ID = 'amsterdamumcdb'
    DATASET_ID = 'version1_5_0'
    LOCATION = 'eu'

    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/home/m/.config/gcloud/application_default_credentials.json'
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

    ### ------------------------------- SETUP Done ------------------------------- ###
    ### -------------------------------------------------------------------------- ###
    df_admissions = pd_gbq.read_gbq(combined_diagnosis_query, project_id=PROJECT_ID, configuration=config_gbq, use_bqstorage_api=True, credentials=bq_client._credentials)
    df_combined_diagnoses = pd_gbq.read_gbq(combined_diagnoses_query, project_id=PROJECT_ID, configuration=config_gbq, use_bqstorage_api=True, credentials=bq_client._credentials)
    
    
    df_sofa_resp = pd_gbq.read_gbq(query_sofa_respiratory, project_id=PROJECT_ID, configuration=config_gbq, use_bqstorage_api=True, credentials=bq_client._credentials)
    df_sofa_resp, df_fio2_cleaned, df_pao2_cleaned = get_sofa_resp(df_sofa_resp)
    
    
    df_sofa_coagulation = pd_gbq.read_gbq(query_sofa_coagulation, project_id=PROJECT_ID, configuration=config_gbq, use_bqstorage_api=True, credentials=bq_client._credentials)
    df_sofa_coagulation = get_sofa_coagulation(df_sofa_coagulation)


    df_sofa_liver = pd_gbq.read_gbq(query_sofa_liver, project_id=PROJECT_ID, configuration=config_gbq, use_bqstorage_api=True, credentials=bq_client._credentials)
    df_sofa_liver = get_sofa_liver(df_sofa_liver)

    df_sofa_cardiovascular_meds = pd_gbq.read_gbq(query_vasopressors_ionotropes, project_id=PROJECT_ID, configuration=config_gbq, use_bqstorage_api=True, credentials=bq_client._credentials)
    df_sofa_cardiocascular_meds = get_cardiovascular_meds(df_sofa_cardiovascular_meds)

    df_blood_pressure = pd_gbq.read_gbq(query_blood_pressure, project_id=PROJECT_ID, configuration=config_gbq, use_bqstorage_api=True, credentials=bq_client._credentials)
    df_blood_pressure = get_blood_pressure(df_blood_pressure)

    #combine the scores from MAP and cardiovascular medication
    df_sofa_cardiovascular = pd.concat([df_blood_pressure, df_sofa_cardiovascular_meds], sort=False).sort_values(by='visit_occurrence_id')


    df_sofa_cns = pd_gbq.read_gbq(query_sofa_cns, project_id=PROJECT_ID, configuration=config_gbq, use_bqstorage_api=True, credentials=bq_client._credentials)
    df_sofa_cns = get_sofa_cns(df_sofa_cns)


    df_sofa_renal = pd_gbq.read_gbq(query_sofa_renal, project_id=PROJECT_ID, configuration=config_gbq, use_bqstorage_api=True, credentials=bq_client._credentials)
    df_sofa_renal = get_sofa_renal(df_sofa_renal)


    df_sofa_creatinine = pd_gbq.read_gbq(query_sofa_creatinine, project_id=PROJECT_ID, configuration=config_gbq, use_bqstorage_api=True, credentials=bq_client._credentials)
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

    df_combined_diagnoses = pd.merge(df_combined_diagnoses, sofa, on='visit_occurrence_id', how='inner')
    df_sepsis_at_admission = df_combined_diagnoses[(
        (df_combined_diagnoses['sepsis_at_admission'] == 1) |
        (
            (df_combined_diagnoses['sepsis_antibiotics_bool'] == 1) &
            (df_combined_diagnoses['sepsis_cultures_bool'] == 1) &
            (df_combined_diagnoses['sofa_total_score'] >= 2)
        )
    )]

    print("INFO: ")
    print(f"Unique persons: {df_sepsis_at_admission['person_id'].unique()}")
    print(f"Unique admissions: {df_sepsis_at_admission['visit_occurrence_id'].unique()}")


    print(df_sofa.head(10))
