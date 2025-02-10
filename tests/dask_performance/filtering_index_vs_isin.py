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


def filter_with_isin_with_index(df, pids):
    """Filter using isin approach with index"""
    start_time = time.time()
    df = df.PID.isin(pids)
    df = df.set_index("PID", drop=True)
    df = df.reset_index(drop=False)
    df = df.compute()
    end_time = time.time()
    return end_time - start_time


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
    df = df.compute()
    end_time = time.time()
    return end_time - start_time


def filter_with_isin(df, pids):
    """Filter using isin approach"""
    start_time = time.time()

    df = df[df["PID"].isin(pids)]
    df = df.compute()
    end_time = time.time()
    return end_time - start_time


def filter_with_loc_if_index_already_set(df, pids):
    """
    Filter using loc if index is already set
    Mease just the contribution of the loc operation
    """
    start_index_time = time.time()
    df = df.set_index("PID", drop=True)
    df = df.reset_index(drop=False)
    df.compute()
    end_index_time = time.time()

    index_loc_time = filter_with_index_loc(df, pids)
    time_diff = index_loc_time - (end_index_time - start_index_time)
    return time_diff


def main(n_runs=5, n_patients=10_000, mean_events_per_patient=10, sample_size=10_000):
    index_loc_times = []
    isin_times = []
    loc_if_index_already_set_times = []
    # Create the test data
    df = create_large_dataframe(
        n_patients=n_patients, mean_events_per_patient=mean_events_per_patient
    )
    df_copy = df.copy(deep=True)
    # Convert to Dask DataFrame

    with ProgressBar():
        for _ in range(n_runs):
            # Test both approaches with same data
            test_pids = np.random.choice(
                df["PID"].unique(), size=sample_size, replace=False
            )
            df_dask = dd.from_pandas(df, npartitions=100)
            df_dask_copy = dd.from_pandas(df_copy, npartitions=100)
            # Test isin approach
            isin_time = filter_with_isin(df_dask_copy, test_pids)
            isin_times.append(isin_time)

            # Test index+loc approach
            index_loc_time = filter_with_index_loc(df_dask, test_pids)
            index_loc_times.append(index_loc_time)

            # Test loc if index is already set
            loc_if_index_already_set_time = filter_with_loc_if_index_already_set(
                df_dask, test_pids
            )
            loc_if_index_already_set_times.append(loc_if_index_already_set_time)

    print(
        f"Average time for set_index + loc approach: {np.mean(index_loc_times):.4f} seconds"
    )

    print(f"Average time for isin approach: {np.mean(isin_times):.4f} seconds")
    print(
        f"Average time for loc if index is already set: {np.mean(loc_if_index_already_set_times):.4f} seconds"
    )


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
    parser.add_argument(
        "--sample-size",
        type=int,
        default=10_000,
        help="Number of PIDs to sample for filtering (default: 10,000)",
    )

    args = parser.parse_args()

    main(
        n_runs=args.n_runs,
        n_patients=args.n_patients,
        mean_events_per_patient=args.events_per_patient,
        sample_size=args.sample_size,
    )
