import pandas as pd
from typing import List

def get_col_booleans(concepts_plus:pd.DataFrame, columns:List, patterns:List[List[str]], 
                    match_how:str='startswith', case_sensitive:bool=True)->list:
    """
    Get boolean columns for each type and match.
    """
    col_booleans = []
    for col, pattern in zip(columns, patterns):
        if match_how=='startswith':
            col_bool = startswith_match(concepts_plus, col, pattern, case_sensitive)
        elif match_how == 'contains':
            col_bool = contains_match(concepts_plus, col, pattern, case_sensitive)
        else:
            raise ValueError(f"match_how must be startswith or contains, not {match_how}")
        col_booleans.append(col_bool)
    return col_booleans

def startswith_match(df: pd.DataFrame, column: str, patterns: List[str], case_sensitive: bool) -> pd.Series:
    """Match strings using startswith"""
    if not case_sensitive:
        patterns = [x.lower() for x in patterns]
        return df[column].astype(str).str.lower().str.startswith(tuple(patterns), False)
    return df[column].astype(str).str.startswith(tuple(patterns), False)

def contains_match(df: pd.DataFrame, column: str, patterns: List[str], case_sensitive: bool) -> pd.Series:
    """Match strings using contains"""
    col_bool = pd.Series([False] * len(df), index=df.index, name=column)
    for pattern in patterns:
        if not case_sensitive:
            pattern = pattern.lower()
        if case_sensitive:
            col_bool |= df[column].astype(str).str.contains(pattern, na=False) 
        else: 
            col_bool |= df[column].astype(str).str.lower().str.contains(pattern, na=False)
    return col_bool