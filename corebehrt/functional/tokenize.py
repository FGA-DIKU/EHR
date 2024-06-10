import polars as pl

def find_visit_change(group):
    """Sort by 'abspos' and filter where 'segment' changes compared to the next entry."""
    return group.sort('abspos').filter(pl.col('segment') != pl.col('segment').shift(-1))

def find_first_event(group):
    """Select the first event in the group."""
    return group.sort('abspos').slice(0, 1)

def add_token(
        features: pl.DataFrame, token: str, 
        group_func: callable, abspos_adjustment: float) -> pl.DataFrame:
    """
    Add a token to the DataFrame.
    First group on PID then apply a function to each group 
    to select the position of the token.
    The abspos_adjustment is used to ensure the tokens will be placed correctly.
    """
    events = features.groupby('PID', maintain_order=True).apply(group_func)
    events = events.with_columns([
        pl.lit(token).alias('concept'),
        (pl.col('abspos') + abspos_adjustment).alias('abspos')
    ])
    features = pl.concat([features, events])
    return features.sort(['PID', 'abspos'])

def add_separator_token(features: pl.DataFrame, sep_token: str = '[SEP]') -> pl.DataFrame:
    """Add separator token after each segment."""
    return add_token(features, sep_token, find_visit_change, 1e-3)

def add_cls_token(features: pl.DataFrame, cls_token: str = '[CLS]') -> pl.DataFrame:
    """Add a classification token to the beginning of each patient's sequence"""
    return add_token(features, cls_token, find_first_event, -1e-3)

