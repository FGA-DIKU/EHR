import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy import stats

def sweep_threshold_auc(data: pd.DataFrame) -> None:
    """
    Sweep through every possible threshold to compute ROC AUC using lab values only.
    Used to compare with theoretical AUC.
    
    Args:
        data: DataFrame with 'code', 'is_positive', and 'numeric_value' columns.
    """
    lab_data = data[data['code'] == 'S/LAB1']
    
    # Extract values and labels
    values = lab_data['numeric_value'].values
    labels = lab_data['is_positive'].astype(int).values
    
    # Optional: Compare with sklearn's built-in AUC (for sanity check)
    sklearn_auc = roc_auc_score(labels, values)
    return sklearn_auc


def manual_mann_whitney_u(data: pd.DataFrame) -> None:
    """
    Calculate the theoretical perfect ROC AUC based on the lab value distributions.
    
    Args:
        data: DataFrame containing the modified data
    """
    # Get lab values for positive and negative patients
    lab_mask = data['code'] == 'S/LAB1'
    positive_values = data[lab_mask & data['is_positive']]['numeric_value']
    negative_values = data[lab_mask & ~data['is_positive']]['numeric_value']
    
    # Calculate theoretical AUC using the Mann-Whitney U statistic
    # AUC = (U + n1 * (n1 + 1) / 2) / (n1 * n2)
    # where U is the number of concordant pairs
    
    n1, n2 = len(positive_values), len(negative_values)
    
    # Count concordant pairs (positive > negative)
    concordant_pairs = 0
    ties = 0
    
    for pos_val in positive_values:
        for neg_val in negative_values:
            if pos_val > neg_val:
                concordant_pairs += 1
            elif pos_val == neg_val:
                ties += 1
    
    # Calculate U statistic
    U = concordant_pairs + 0.5 * ties
    
    # Calculate theoretical AUC
    theoretical_auc = U / (n1 * n2)
    return theoretical_auc

def scipy_mann_whitney_u(data: pd.DataFrame) -> None:
    """
    Calculate the theoretical perfect ROC AUC based on the lab value distributions.
    
    Args:
        data: DataFrame containing the modified data
    """
    lab_mask = data['code'] == 'S/LAB1'
    positive_values = data[lab_mask & data['is_positive']]['numeric_value']
    negative_values = data[lab_mask & ~data['is_positive']]['numeric_value']
    n1, n2 = len(positive_values), len(negative_values)
    statistic, p_value = stats.mannwhitneyu(positive_values, negative_values, alternative='greater')
    auc_from_scipy = statistic / (n1 * n2)
    return auc_from_scipy

def cohens_d(data: pd.DataFrame) -> None:
    """
    Calculate the Cohen's d statistic for the lab value distributions.
    
    Args:
        data: DataFrame containing the modified data
    """
    lab_mask = data['code'] == 'S/LAB1'
    positive_values = data[lab_mask & data['is_positive']]['numeric_value']
    negative_values = data[lab_mask & ~data['is_positive']]['numeric_value']
    n1, n2 = len(positive_values), len(negative_values)

    # Calculate overlap between distributions
    pos_mean, pos_std = positive_values.mean(), positive_values.std()
    neg_mean, neg_std = negative_values.mean(), negative_values.std()
    
    # Calculate separation metric (Cohen's d)
    pooled_std = np.sqrt(((n1 - 1) * pos_std**2 + (n2 - 1) * neg_std**2) / (n1 + n2 - 2))
    cohens_d = (pos_mean - neg_mean) / pooled_std
    return cohens_d