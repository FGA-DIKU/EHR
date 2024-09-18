import dask.dataframe as dd
import pandas as pd
import operator
import dask.dataframe as dd



def censor_data(data: dd.DataFrame, censor_dates: pd.Series)-> dd.DataFrame:
    """
    Censors the data by removing all events that occur after the censor_dates.
    args:
        data: dd.DataFrame (needs to have abspos column)
        censor_dates: pd.Series (index: PID, values: censor_dates as abspos)
    """
    return filter_events_by_abspos(data, censor_dates, '<=')
    
def filter_events_by_abspos(
    data: dd.DataFrame,
    abspos_series: pd.Series,
    comparison_operator: str,
) -> dd.DataFrame:
    """
    Filters the data based on a timestamp per PID using the specified comparison operator.

    Args:
        data: DataFrame with 'PID' and 'abspos' columns.
        abspos_series: Series with index 'PID' and values as abspos.
        comparison_operator: A string representing the comparison operator, e.g., '<=', '>=', '<', '>'.

    Returns:
        The filtered DataFrame.
    """

    comp_func = get_comparison_function(comparison_operator)

    # Convert the Series to a DataFrame
    abspos_df = abspos_series.reset_index()
    abspos_df.columns = ['PID', 'abspos_ref']

    merged_df = dd.merge(data, abspos_df, on='PID', how='inner')
    filtered_df = merged_df[comp_func(merged_df['abspos'], merged_df['abspos_ref'])]
    
    return filtered_df.drop(columns=['abspos_ref'])

def get_comparison_function(comparison_operator: str) -> None:
    """Map the string comparison_operator to an actual operator function"""
    operators = {
        '<=': operator.le,
        '>=': operator.ge,
        '<': operator.lt,
        '>': operator.gt,
        '==': operator.eq,
        '!=': operator.ne
    }

    if comparison_operator not in operators:
        raise ValueError(f"Invalid comparison_operator '{comparison_operator}'. Must be one of {list(operators.keys())}.")
    return operators[comparison_operator]