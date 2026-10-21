import operator as op_

import polars as pl

COMPARE = {"lt": op_.lt, "le": op_.le, "gt": op_.gt, "ge": op_.ge, "eq": op_.eq}
DEPENDENCES = {"any", "all", "all_row"}


def get_condition_expression(condition) -> pl.Expr:
    dependence = DEPENDENCES.intersection(condition)
    if dependence:
        if len(dependence) != 1:
            raise ValueError("A criteria group must have exactly one dependence")
        dep = dependence.pop()
        exprs = [get_condition_expression(c) for c in condition[dep]]
        if not exprs:
            raise ValueError(f"Dependence={dep} requires at least one condition")
        if dep == "any":
            return pl.any_horizontal(exprs)
        if dep == "all_row":
            return pl.all_horizontal(exprs)

        time = pl.col("time")
        return pl.any_horizontal(exprs) & pl.all_horizontal(
            time >= time.filter(expr).min().over("subject_id") for expr in exprs
        )

    op = condition["operator"]
    col = pl.col(condition["column"])
    if op == "in":
        return col.is_in(condition["value"])
    if op == "startswith_any":
        return pl.any_horizontal([col.str.starts_with(v) for v in condition["value"]])
    if op in COMPARE:
        if "value_col" in condition:
            rhs = pl.col(condition["value_col"])
        elif "date_format" in condition:
            rhs = pl.lit(condition["value"]).str.to_date(condition["date_format"])
        else:
            rhs = pl.lit(condition["value"])
        if condition.get("offset_by"):
            rhs = rhs.dt.offset_by(condition["offset_by"])
        return COMPARE[op](col, rhs)
    raise ValueError(f"Operator={op} not supported")


def get_matches(df: pl.DataFrame, args) -> pl.DataFrame:
    """Return all times each subject meets an inline criteria definition."""
    occurence = args["occurence"]
    criteria = {key: value for key, value in args.items() if key != "occurence"}
    if len(criteria) != 1:
        raise ValueError("Match args must contain exactly one criteria group")

    expression = get_condition_expression(criteria)
    matches = df.filter(expression).select("subject_id", "time").unique()

    if occurence == "first":
        matches = matches.group_by("subject_id").agg(pl.col("time").min())
    elif occurence == "last":
        matches = matches.group_by("subject_id").agg(pl.col("time").max())
    elif occurence == "all":
        pass
    else:
        raise ValueError(f"Occurence={occurence} not supported")

    return matches
