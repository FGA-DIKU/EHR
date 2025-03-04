"""
Simulate large data for performance testing and profiling.
"""

import os

import numpy as np
import pandas as pd
import argparse

import pyarrow.parquet as pq
from pandas import NaT

DEFAULT_READ_DIR = "tmp/example_data"
DEFAULT_WRITE_DIR = "tmp/example_MEDS_data"
DEFAULT_BATCH_SIZE = 200_000
DEFAULT_SHARD_SIZE = 500
DEFAULT_SPLIT = {"train": 0.8, "tuning": 0.1, "held_out": 0.1}
DEFAULT_DOB = "BIRTHDATE"
DEFAULT_DOD = "DEATHDATE"
DEFAULT_BG = ["RACE", "ETHNICITY", "GENDER"]


def get_bg_for_shard(shard_pids, patients_info, DOB, DOD, BG_cols):
    subject_ids = []
    time_list = []
    code_list = []
    numeric_value_list = []

    shard_patients_info = patients_info[patients_info["subject_id"].isin(shard_pids)]

    # DOB
    pids = shard_patients_info["subject_id"].tolist()
    for pid in pids:
        dob = patients_info[patients_info["subject_id"] == pid][DOB].values[0]
        time_list.extend([dob])
        subject_ids.extend([pid])
        code_list.extend(["DOB"])
        numeric_value_list.extend([np.nan])

    # BG
    for col in BG_cols:
        code = [f"{col}//{val}" for val in shard_patients_info[col]]

        subject_ids.extend(pids)
        time_list.extend([NaT] * len(pids))
        code_list.extend(code)
        numeric_value_list.extend([np.nan] * len(pids))

    # DOD if available
    dod_pids = shard_patients_info[~shard_patients_info[DOD].isna()][
        "subject_id"
    ].tolist()
    for pid in dod_pids:
        dod = patients_info[patients_info["subject_id"] == pid][DOD].values[0]
        time_list.extend([dod])
        subject_ids.extend([pid])
        code_list.extend(["DOD"])
        numeric_value_list.extend([np.nan])

    df = pd.DataFrame(
        {
            "subject_id": subject_ids,
            "time": time_list,
            "code": code_list,
            "numeric_value": numeric_value_list,
        }
    )

    df["time"] = pd.to_datetime(df["time"])
    return df


def get_concepts_for_shard(read_dir, shard_pids, hash_to_integer_map, patients_info):
    subject_ids = []
    time = []
    code = []
    numeric_value = []

    # Getting conepts
    for concept_type in os.listdir(read_dir):
        if concept_type == "patients_info.parquet":
            continue
        if concept_type.endswith(".parquet"):
            concept = pq.read_table(f"{read_dir}/{concept_type}").to_pandas()
        elif concept_type.endswith(".csv"):
            concept = pd.read_csv(f"{read_dir}/{concept_type}")
        else:
            raise ValueError(f"Unsupported file type: {concept_type}")
        concept["subject_id"] = concept["PID"].map(hash_to_integer_map)
        concept = concept[concept["subject_id"].isin(shard_pids)]
        subject_ids.extend(concept["subject_id"])
        time.extend(concept["TIMESTAMP"])
        code.extend(concept["CONCEPT"])
        numeric_value.extend(
            concept["RESULT"].round(2)
            if "RESULT" in concept.columns
            else [np.nan] * len(concept)
        )

    return pd.DataFrame(
        {
            "subject_id": subject_ids,
            "time": time,
            "code": code,
            "numeric_value": numeric_value,
        }
    )


def main_write(read_dir, write_dir, shard_size, split, patient_info_args):
    patients_info = pq.read_table(f"{read_dir}/patients_info.parquet").to_pandas()
    patients_info["subject_id"], unique_values = pd.factorize(patients_info["PID"])
    patients_info["subject_id"] = patients_info["subject_id"].astype(int)
    hash_to_integer_map = dict(zip(unique_values, range(len(unique_values))))
    patients_info = patients_info.drop(columns=["PID"])

    pids = patients_info["subject_id"].unique().tolist()
    n = len(pids)
    pretrain_pids = pids[: int(n * split["train"])]
    finetune_pids = pids[
        int(n * split["train"]) : int(n * (split["train"] + split["tuning"]))
    ]
    test_pids = pids[int(n * (split["train"] + split["tuning"])) :]

    shard_dict = {}

    for split_name, split_pids in zip(
        ["train", "tuning", "held_out"], [pretrain_pids, finetune_pids, test_pids]
    ):
        shard_dict[split_name] = []
        counter = 0
        for i in range(0, len(split_pids), shard_size):
            shard_pids = split_pids[i : i + shard_size]
            shard_concept = get_concepts_for_shard(
                read_dir, shard_pids, hash_to_integer_map, patients_info
            )
            shard_bg = get_bg_for_shard(shard_pids, patients_info, **patient_info_args)
            complete_shard = pd.concat([shard_concept, shard_bg], axis=0).sort_values(
                by=["subject_id", "time"]
            )
            complete_shard = complete_shard.reset_index(drop=True)
            os.makedirs(f"{write_dir}/{split_name}", exist_ok=True)
            complete_shard.to_parquet(
                f"{write_dir}/{split_name}/{counter}.parquet", index=True
            )
            counter += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Simulate MEDS like data (without metadata)"
    )

    parser.add_argument(
        "--read-dir",
        type=str,
        default=DEFAULT_READ_DIR,
        help=f"Directory to read input files (default: {DEFAULT_READ_DIR})",
    )

    parser.add_argument(
        "--write-dir",
        type=str,
        default=DEFAULT_WRITE_DIR,
        help=f"Directory to write output files (default: {DEFAULT_WRITE_DIR})",
    )

    parser.add_argument(
        "--shard-size",
        type=int,
        default=DEFAULT_SHARD_SIZE,
        help=f"Size of each shard (default: {DEFAULT_SHARD_SIZE})",
    )

    parser.add_argument(
        "--split",
        type=dict,
        default=DEFAULT_SPLIT,
        help=f"Split of data into pretrain, finetune, test (default: {DEFAULT_SPLIT})",
    )

    parser.add_argument(
        "--dob",
        type=str,
        default=DEFAULT_DOB,
        help=f"DOB column name (default: {DEFAULT_DOB})",
    )

    parser.add_argument(
        "--dod",
        type=str,
        default=DEFAULT_DOD,
        help=f"DOD column name (default: {DEFAULT_DOD})",
    )

    parser.add_argument(
        "--bg",
        type=list,
        default=DEFAULT_BG,
        help=f"Background columns (default: {DEFAULT_BG})",
    )

    args = parser.parse_args()
    main_write(
        read_dir=args.read_dir,
        write_dir=args.write_dir,
        shard_size=args.shard_size,
        split=args.split,
        patient_info_args={"DOB": args.dob, "DOD": args.dod, "BG_cols": args.bg},
    )
