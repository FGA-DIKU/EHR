import argparse
import time

import dask.dataframe as dd
import numpy as np
from dask.diagnostics import ProgressBar
from generate_features import create_large_dataframe


def filter_with_index_loc(df, pids):
    """Filter using set_index + loc approach"""
    start_time = time.time()

    df = df.set_index("PID", drop=True)
    df = df.loc[pids]
    df = df.reset_index(drop=False)

    end_time = time.time()
    return end_time - start_time


def filter_with_isin(df, pids):
    """Filter using isin approach"""
    start_time = time.time()

    df = df[df["PID"].isin(pids)]

    end_time = time.time()
    return end_time - start_time


def main(n_runs=5, n_patients=10_000, mean_events_per_patient=10):
    index_loc_times = []
    isin_times = []

    # Create the test data
    df = create_large_dataframe(
        n_patients=n_patients, mean_events_per_patient=mean_events_per_patient
    )
    # Convert to Dask DataFrame
    df = dd.from_pandas(df, npartitions=100)
    # Test both approaches with same data
    test_pids = np.random.choice(df["PID"].compute().unique(), size=1000, replace=False)

    with ProgressBar():
        for _ in range(n_runs):
            # Create fresh copy of dataframe for each test
            df_copy = df.copy()

            # Test isin approach
            isin_time = filter_with_isin(df_copy, test_pids)
            isin_times.append(isin_time)

            # Create fresh copy of dataframe for each test
            df_copy = df.copy()

            # Test index+loc approach
            index_loc_time = filter_with_index_loc(df_copy, test_pids)
            index_loc_times.append(index_loc_time)

    print(
        f"Average time for set_index + loc approach: {np.mean(index_loc_times):.4f} seconds"
    )
    print(f"Average time for isin approach: {np.mean(isin_times):.4f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare filtering performance with different methods"
    )
    parser.add_argument(
        "--n-patients",
        type=int,
        default=10_000,
        help="Number of unique patients (default: 10,000)",
    )
    parser.add_argument(
        "--events-per-patient",
        type=int,
        default=10,
        help="Mean number of events per patient (default: 10)",
    )
    parser.add_argument(
        "--n-runs", type=int, default=5, help="Number of test runs (default: 5)"
    )

    args = parser.parse_args()

    main(
        n_runs=args.n_runs,
        n_patients=args.n_patients,
        mean_events_per_patient=args.events_per_patient,
    )
