import pandas as pd
import numpy as np



def calculate_actions(df_sequences, df_vasopressors):
    # Step 2: Apply the function to df2
    gamma_sum = df_sequences.apply(lambda row: compute_window_gamma(row, df_vasopressors), axis=1)

    # Calculate average gamma (handle total_hours = 0 case)
    gamma_sum["average_gamma"] = gamma_sum["total_gamma"] / gamma_sum['total_hours']
    gamma_sum['average_gamma'] = gamma_sum['average_gamma'].fillna(0)  # Fill NaN for empty windows
    return gamma_sum["average_gamma"]


def compute_window_gamma(row, df_vasopressors):
    # Filter intervals from df1 that overlap with the current window
    overlapping = df_vasopressors[
        (df_vasopressors['start_time'] < row['window_end']) &\
        (df_vasopressors['end_time'] > row['window_start'])
    ]
    
    if overlapping.empty:
        return pd.Series({'total_gamma': 0, 'total_hours': 0})
    
    # Calculate overlap duration for normalization
    overlapping['overlap_start'] = overlapping[['start_time', 'window_start']].max(axis=1)
    overlapping['overlap_end'] = overlapping[['end_time', 'window_end']].min(axis=1)
    overlapping['overlap_duration'] = (overlapping['overlap_end'] - overlapping['overlap_start']).dt.total_seconds() / 3600  # in hours
    
    # Compute weighted gamma
    overlapping['weighted_gamma'] = overlapping['gamma'] * overlapping['overlap_duration']
    
    # Aggregate results for this window
    total_gamma = overlapping['weighted_gamma'].sum()
    total_hours = overlapping['overlap_duration'].sum()
    return pd.Series({'total_gamma': total_gamma, 'total_hours': total_hours})
