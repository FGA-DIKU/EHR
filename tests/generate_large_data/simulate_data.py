import os
import uuid
import numpy as np
import pandas as pd
from tqdm import tqdm

N = 100_000
BATCH_SIZE = 10_000
AVERAGE_CONCEPTS = 10  # Average number of concepts per patient
WRITE_DIR = "tmp/example_data_large_100k"


def main_write(
    n_patients=10000,
    batch_size_patients=10000,
    average_concepts=10,
    write_dir="../example_data_large",
):
    os.makedirs(write_dir, exist_ok=True)
    for i in tqdm(range(n_patients // batch_size_patients)):
        patients_info = generate_patients_info_batch(batch_size_patients)
        patients_info.to_csv(
            f"{write_dir}/patients_info.csv",
            index=False,
            mode="w" if i == 0 else "a",
            header=i == 0,
        )

        # Generate variable number of concepts per patient
        n_records_per_pid_list = np.ceil(
            np.random.exponential(scale=average_concepts, size=batch_size_patients)
        ).astype(int)
        n_records_per_pid_list[n_records_per_pid_list == 0] = (
            1  # Ensure at least 1 concept
        )

        # Generate concepts for diagnoses
        concepts = generate_concepts_batch(patients_info, n_records_per_pid_list)
        concepts.to_csv(
            f"{write_dir}/concept.diagnose.csv",
            index=False,
            mode="w" if i == 0 else "a",
            header=i == 0,
        )

        # Generate concepts for medications (optional: you can use a different distribution)
        n_records_per_pid_list_m = np.ceil(
            np.random.exponential(scale=average_concepts, size=batch_size_patients)
        ).astype(int)
        n_records_per_pid_list_m[n_records_per_pid_list_m == 0] = 1

        concepts_m = generate_concepts_batch(patients_info, n_records_per_pid_list_m)
        concepts_m.to_csv(
            f"{write_dir}/concept.medication.csv",
            index=False,
            mode="w" if i == 0 else "a",
            header=i == 0,
        )
        concepts_l = generate_concepts_batch(
            patients_info,
            n_concepts,
            prefix="LAB",
            result_col=True,
            n_unique_concepts=10,
        )
        concepts_l.to_csv(
            f"{write_dir}/concept.labtest.csv",
            index=False,
            mode="w" if i == 0 else "a",
            header=i == 0,
        )


def generate_patients_info_batch(n_patients):
    np.random.seed(42)

    start_birthdate = np.datetime64("1940-01-01")
    end_birthdate = np.datetime64("2020-01-01")

    birthdates = np.random.choice(
        np.arange(start_birthdate, end_birthdate, dtype="datetime64[D]"), n_patients
    )

    death_prob = np.random.rand(n_patients)

    deathdates = np.where(
        death_prob > 0.8,
        np.array(
            [
                np.random.choice(
                    np.arange(
                        birthdate + np.timedelta64(10),
                        np.datetime64("2024-01-01"),
                        dtype="datetime64[D]",
                    )
                )
                for birthdate in birthdates
            ]
        ),
        pd.NaT,
    )

    races = np.random.choice(
        ["white", "black", "asian", "native", "other"], size=n_patients
    )
    ethnicities = np.random.choice(["hispanic", "nonhispanic"], size=n_patients)
    genders = np.random.choice(["M", "F"], size=n_patients)

    pids = np.array([str(uuid.uuid4()) for _ in range(n_patients)])

    return pd.DataFrame(
        {
            "PID": pids,
            "BIRTHDATE": birthdates,
            "DEATHDATE": deathdates,
            "RACE": races,
            "ETHNICITY": ethnicities,
            "GENDER": genders,
        }
    )


def generate_concepts_batch(patients_info, n_records_per_pid_list):
    # Repeat each patient according to the number of concepts
    repeated_patients_info = patients_info.loc[
        patients_info.index.repeat(n_records_per_pid_list)
    ].reset_index(drop=True)

    repeated_patients_info["BIRTHDATE"] = pd.to_datetime(
        repeated_patients_info["BIRTHDATE"]
    )
    end_dates = pd.to_datetime(repeated_patients_info["DEATHDATE"]).fillna(
        pd.Timestamp(year=2025, month=1, day=1)
    )

    birthdates = repeated_patients_info["BIRTHDATE"].astype(np.int64) // 10**9
    deathdates = end_dates.astype(np.int64) // 10**9
    valid_mask = birthdates < deathdates
    deathdates[~valid_mask] = birthdates[~valid_mask] + 1

    random_timestamps = np.random.randint(birthdates, deathdates, dtype=np.int64)
    timestamps = pd.to_datetime(random_timestamps, unit="s")

    admission_ids = np.array(
        [str(uuid.uuid4()) for _ in range(len(repeated_patients_info))]
    )

    concepts = np.random.randint(100000000, 999999999, size=len(repeated_patients_info))

    concepts_data = pd.DataFrame(
        {
            "TIMESTAMP": timestamps,
            "PID": repeated_patients_info["PID"],
            "ADMISSION_ID": admission_ids,
            "CONCEPT": concepts,
        }
    )

    if result_col:
        results = np.random.randint(100, 200, size=len(repeated_patients_info))
        concepts_data["RESULT"] = results

    # Filter out rows where TIMESTAMP is less than BIRTHDATE
    concepts_data = concepts_data[
        concepts_data["TIMESTAMP"] >= repeated_patients_info["BIRTHDATE"].values
    ]

    return concepts_data


if __name__ == "__main__":
    main_write(
        n_patients=N,
        batch_size_patients=BATCH_SIZE,
        average_concepts=AVERAGE_CONCEPTS,
        write_dir=WRITE_DIR,
    )
