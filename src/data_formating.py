import pandas as pd
from tqdm import tqdm


def resample_group(group_data: pd.api.typing.DataFrameGroupBy, index_col: str, window_length="4h"):
    group_data = group_data.set_index(index_col)
    df_resampled = group_data.resample(window_length).mean()
    return df_resampled


def transform_long_to_wide_sepsis(
        df_data_long: pd.DataFrame,
        sort_column="measurement_datetime",
        pivot_index=["person_id", "visit_occurrence_id", "measurement_datetime"],
        feature_name_column='feature_name',
        value_column='value_as_number',
        group_columns=["person_id", "visit_occurrence_id"]
    ):
    df_data_long = df_data_long.sort_values(by=sort_column)
    df_data_wide = df_data_long.pivot(index=pivot_index, columns=feature_name_column, values=value_column)
    df_data_wide = df_data_wide.reset_index()
    # this is needed to display progress in the progress_apply function
    tqdm.pandas()
    data_wide_grouped = df_data_wide.groupby(group_columns)
    df_data_wide_resampled = data_wide_grouped.progress_apply(resample_group, args=("measurement_datetime"))
    df_data_wide_resampled = df_data_wide_resampled.reset_index(level=["person_id", "visit_occurrence_id"], drop=True)
    #df_data_wide_resampled = df_data_wide_resampled.reset_index()
    df_data_wide_resampled["seq_id"] = df_data_wide_resampled.groupby(group_columns).cumcount()
    return df_data_wide_resampled
