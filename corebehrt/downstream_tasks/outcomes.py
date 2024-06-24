from typing import Dict, List

import numpy as np
import pandas as pd

from corebehrt.data.utils import Utilities


class OutcomeMaker:
    def __init__(self, config: dict, features_cfg: dict):
        self.outcomes = config.outcomes
        self.features_cfg = features_cfg
        self.config = config

    def __call__(
        self, concepts_plus: pd.DataFrame, patients_info: pd.DataFrame
    )->dict:
        """Create outcomes from concepts_plus and patients_info"""
        patient_set = concepts_plus.PID.unique()
        concepts_plus = self.filter_table_by_pids(concepts_plus, patient_set)
        patients_info = self.filter_table_by_pids(patients_info, patient_set)
        concepts_plus = self.remove_missing_timestamps(concepts_plus)
 
        outcome_tables = {}
        for outcome, attrs in self.outcomes.items():
            types = attrs["type"]
            matches = attrs["match"]
            if types == "patients_info":
                timestamps = self.match_patient_info(patients_info, matches)
            else:
                timestamps = self.match_concepts(concepts_plus, types, matches, attrs)
            timestamps['TIMESTAMP'] = Utilities.get_abspos_from_origin_point(timestamps['TIMESTAMP'], self.features_cfg.features.origin_point) 
            timestamps['TIMESTAMP'] = timestamps['TIMESTAMP'].astype(int)
            outcome_tables[outcome] = timestamps
        return outcome_tables

    @staticmethod
    def filter_table_by_pids(table: pd.DataFrame, pids: List[str])->pd.DataFrame:
        return table[table.PID.isin(pids)]
    
    @staticmethod
    def remove_missing_timestamps(concepts_plus: pd.DataFrame )->pd.DataFrame:
        return concepts_plus[concepts_plus.TIMESTAMP.notna()]

    def match_patient_info(self, patients_info: dict, match: List[List])->pd.Series:
        """Get timestamps of interest from patients_info"""
        return patients_info[['PID', match]].dropna()

    def match_concepts(self, concepts_plus: pd.DataFrame, types: List[List], 
                       matches:List[List], attrs:Dict)->pd.DataFrame:
        """It first goes through all the types and returns true for a row if the entry starts with any of the matches.
        We then ensure all the types are true for a row by using bitwise_and.reduce. E.g. CONCEPT==COVID_TEST AND VALUE==POSITIVE"""
        if 'exclude' in attrs:
            concepts_plus = concepts_plus[~concepts_plus['CONCEPT'].isin(attrs['exclude'])]
        col_booleans = self.get_col_booleans(concepts_plus, types, matches, 
                                             attrs.get("match_how", 'startswith'), attrs.get("case_sensitive", True))
        mask = np.bitwise_and.reduce(col_booleans)
        if "negation" in attrs:
            mask = ~mask
        return concepts_plus[mask].drop(columns=['ADMISSION_ID', 'CONCEPT'])
    
    @staticmethod
    def get_col_booleans(concepts_plus:pd.DataFrame, types:List, matches:List[List], 
                         match_how:str='startswith', case_sensitive:bool=True)->list:
        col_booleans = []
        for typ, lst in zip(types, matches):
            if match_how=='startswith':
                col_bool = OutcomeMaker.startswith_match(concepts_plus, typ, lst, case_sensitive)
            elif match_how == 'contains':
                col_bool = OutcomeMaker.contains_match(concepts_plus, typ, lst, case_sensitive)
            else:
                raise ValueError(f"match_how must be startswith or contains, not {match_how}")
            col_booleans.append(col_bool)
        return col_booleans
    
    @staticmethod
    def startswith_match(df: pd.DataFrame, column: str, patterns: List[str], case_sensitive: bool) -> pd.Series:
        """Match strings using startswith"""
        if not case_sensitive:
            patterns = [x.lower() for x in patterns]
            return df[column].astype(str).str.lower().str.startswith(tuple(patterns), False)
        return df[column].astype(str).str.startswith(tuple(patterns), False)
    
    @staticmethod
    def contains_match(df: pd.DataFrame, column: str, patterns: List[str], case_sensitive: bool) -> pd.Series:
        """Match strings using contains"""
        col_bool = pd.Series([False] * len(df), index=df.index)
        for pattern in patterns:
            if not case_sensitive:
                pattern = pattern.lower()
            if case_sensitive:
                col_bool |= df[column].astype(str).str.contains(pattern, na=False) 
            else: 
                col_bool |= df[column].astype(str).str.lower().str.contains(pattern, na=False)
        return col_bool
