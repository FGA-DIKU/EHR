import operator as op_
from typing import Literal

import polars as pl

COMPARE = {"lt": op_.lt, "le": op_.le, "gt": op_.gt, "ge": op_.ge, "eq": op_.eq}


def get_condition_expression(cond) -> pl.Expr:
    op = cond["operator"]
    col = pl.col(cond["column"])
    if op == "in":
        return col.is_in(cond["value"])
    if op == "startswith_any":
        return pl.any_horizontal([col.str.starts_with(v) for v in cond["value"]])
    if op in COMPARE:
        rhs = (
            pl.col(cond["value_col"]) if "value_col" in cond else pl.lit(cond["value"])
        )
        if cond.get("offset"):
            rhs = rhs.dt.offset_by(cond["offset"])
        return COMPARE[op](col, rhs)
    raise ValueError(f"Operator={op} not supported")


def get_matches(
    df: pl.DataFrame,
    conditions: list,
    dependence: Literal["any", "all", "all_row"],
    occurence: Literal["first", "last", None] = None,
) -> pl.DataFrame:
    """All times each subject meets the definition."""
    cols = [f"_c{i}" for i in range(len(conditions))]
    exprs = [get_condition_expression(c) for c in conditions]

    # Finds all matches
    matches = df.select(
        "subject_id", "time", *[expr.alias(col) for col, expr in zip(cols, exprs)]
    ).filter(pl.any_horizontal(cols))

    # Filter based on dependence
    if dependence == "all_row":
        # Finds where all matches
        matches = matches.filter(pl.all_horizontal(cols))
    elif dependence == "all":
        # Finds where all conditions are met
        matches = matches.filter(
            pl.all_horizontal(
                pl.col("time")
                >= pl.col("time").filter(pl.col(c)).min().over("subject_id")
                for c in cols
            )
        )
    elif dependence == "any":
        # Already dropped in `.filter(pl.any_horizontal(cols))` above
        pass
    else:
        raise ValueError(f"Dependence={dependence} not supported")

    matches = matches.select("subject_id", "time").unique()

    if occurence == "first":
        matches = matches.group_by("subject_id").agg(pl.col("time").min())
    elif occurence == "last":
        matches = matches.group_by("subject_id").agg(pl.col("time").max())
    elif occurence == "all":
        pass
    else:
        raise ValueError(f"Occurence={occurence} not supported")

    return matches
