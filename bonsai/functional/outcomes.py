import polars as pl

from bonsai.functional.features import compute_abspos
from bonsai.functional.matching import get_matches


def fill_nans_with_sampled(dates, seed=None):
    if dates.is_null().all():
        raise ValueError("No non-NaN indexing dates found")

    return dates.fill_null(
        dates.drop_nulls().sample(dates.len(), with_replacement=True, seed=seed)
    )


def get_index_dates(df: pl.DataFrame, cfg) -> pl.DataFrame:
    """Index dates for one shard. Supports match/date for index_date and drop/sample for nulls"""
    if cfg.mode == "match":
        index_dates = get_matches(
            df, cfg.args.conditions, cfg.args.dependence, cfg.args.occurence
        )
    elif cfg.mode == "date":
        index_dates = (
            df.select("subject_id").unique().with_columns(time=pl.datetime(**cfg.args))
        )
    else:
        raise ValueError(f"Mode={cfg.mode} not supported")

    index_dates = index_dates.rename({"time": "index_date"})

    if cfg.missing == "drop":
        return index_dates
    elif cfg.missing == "sample":
        index_dates = (
            df.select("subject_id")
            .unique()
            .join(index_dates, on="subject_id", how="left")
            .sort("subject_id")  # fixed order -> reproducible sampling
        )
        return index_dates.with_columns(
            index_date=fill_nans_with_sampled(index_dates["index_date"], seed=42)
        )
    else:
        raise ValueError(f"Missing={cfg.missing} not supported")


def get_outcome_dates(df: pl.DataFrame, cfg, refs: dict) -> pl.DataFrame:
    """Outcome dates for one shard. Supports only 'match'"""
    if cfg.mode == "match":
        outcome_dates = get_matches(
            df, cfg.args.conditions, cfg.args.dependence, cfg.args.occurence
        )
    else:
        raise ValueError(f"Mode={cfg.mode} not supported")

    outcome_dates = outcome_dates.rename({"time": "outcome_date"})

    # Take only rows where outcome_date happens AFTER index_date
    outcome_dates = outcome_dates.join(
        refs["index_date"], on="subject_id", how="left"
    ).filter(pl.col("outcome_date") >= pl.col("index_date"))

    return outcome_dates


def get_censor_dates(df: pl.DataFrame, cfg, refs) -> pl.DataFrame:
    if cfg.mode == "offset":
        censor_dates = refs["index_date"].dt.offset_by(cfg.args.offset_by)
    else:
        raise ValueError(f"Mode={cfg.mode} not supported")

    return censor_dates


def get_cohort(df: pl.DataFrame, cfg, refs) -> pl.DataFrame:
    df = df.join(refs["index_date"], on="subject_id", how="left")

    cohort = df.select("subject_id").unique()
    for cond in cfg.conditional_criteria:
        pop = get_matches(
            df, cond.args.conditions, cond.args.dependence, cond.args.occurence
        )

        if cond.action == "exclude":
            cohort = cohort.join(pop, on="subject_id", how="anti")
        elif cond.action == "include":
            cohort = cohort.join(pop, on="subject_id", how="semi")
        else:
            raise ValueError(f"Action={cond.action} not supported")

    return df


def binarize_outcomes(
    outcomes: pl.DataFrame,
    start_include: dict,
    end_include: dict | None = None,
) -> pl.DataFrame:
    window_start = pl.col("index_date") + pl.duration(**start_include)
    has_outcome = pl.col("outcome_date").is_not_null()
    outcomes = outcomes.filter(~(has_outcome & (pl.col("outcome_date") < window_start)))

    if outcomes.select((pl.col("censor_date") > window_start).any()).item():
        raise ValueError(
            "censor_date is after the prediction window start; outcomes would leak into the input"
        )

    in_window = has_outcome
    if end_include is not None:
        in_window &= pl.col("outcome_date") <= pl.col("index_date") + pl.duration(
            **end_include
        )
    return outcomes.with_columns(label=in_window.cast(pl.Int64))


def split_outcomes(
    outcomes: pl.DataFrame,
    train_key: str,
    val_key: str,
    test_key: str,
):
    return (
        outcomes.filter(pl.col("split") == split_key)
        for split_key in (train_key, val_key, test_key)
    )


def split_and_binarize_outcomes(
    outcomes: pl.DataFrame,
    train_key: str,
    val_key: str,
    test_key: str,
    start_include: dict,
    end_include: dict | None = None,
):
    splits = split_outcomes(outcomes, train_key, val_key, test_key)

    return (
        finalize_outcomes(
            binarize_outcomes(
                split,
                start_include,
                end_include,
            )
        )
        for split in splits
    )


def finalize_outcomes(outcomes: pl.DataFrame) -> dict[int, dict]:
    outcomes = outcomes.with_columns(
        censor_abspos=compute_abspos(pl.col("censor_date"))
    )
    return {
        row["subject_id"]: {
            "label": row["label"],
            "censor_abspos": row["censor_abspos"],
        }
        for row in outcomes.to_dicts()
    }
