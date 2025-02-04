from corebehrt.functional.constants import PID_COL, ABSPOS_COL
import pandas as pd


def get_binary_outcomes(
    index_dates: pd.DataFrame,
    outcomes: pd.DataFrame,
    n_hours_start_follow_up: float = 0,
    n_hours_end_follow_up: float = None,
) -> pd.Series:
    """Get binary outcomes for each patient.

    Args:
        index_dates: DataFrame with PID_COL and abspos columns
        outcomes: DataFrame with PID_COL and abspos columns
        n_hours_start_follow_up: Hours after index date to start follow-up
        n_hours_end_follow_up: Hours after index date to end follow-up (None for no end)

    Returns:
        Series with PID index and int (0 or 1) values indicating if outcome occurred in window
    """

    # Create a mask for outcomes within the follow-up window
    merged = pd.merge(
        outcomes[[PID_COL, ABSPOS_COL]],
        index_dates[[PID_COL, ABSPOS_COL]].rename(columns={ABSPOS_COL: "index_abspos"}),
        on=PID_COL,
    )

    # Calculate relative position from index date
    merged["rel_pos"] = merged[ABSPOS_COL] - merged["index_abspos"]

    # Check if outcome is within window
    in_window = merged["rel_pos"] >= n_hours_start_follow_up
    if n_hours_end_follow_up is not None:
        in_window &= merged["rel_pos"] <= n_hours_end_follow_up

    # Group by patient and check if any outcome is within window
    has_outcome = merged[in_window].groupby(PID_COL).size() > 0

    # Ensure all patients from index_dates are included with False for those without outcomes
    result = pd.Series(
        False, index=index_dates[PID_COL].unique(), name="has_outcome", dtype=bool
    )
    result.index_name = PID_COL
    result[has_outcome.index] = has_outcome
    return result.astype(int)
