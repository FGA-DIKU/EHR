import pandas as pd
import numpy as np


# Simulate data
def create_large_dataframe(n_patients, mean_events_per_patient):
    np.random.seed(0)

    # Generate random number of events for each patient using exponential distribution
    events_per_patient = np.random.exponential(mean_events_per_patient, size=n_patients)
    events_per_patient = np.maximum(
        1, events_per_patient.astype(int)
    )  # Ensure at least 1 event

    # Generate patient IDs with varying repetitions
    pids = np.repeat(np.arange(n_patients), events_per_patient)

    # Generate remaining data based on actual number of events
    ages = np.random.randint(20, 90, size=len(pids))
    events = np.random.choice(["A", "B", "C", "D", "E"], size=len(pids))
    timestamps = np.random.randint(1_000_000, 2_000_000, size=len(pids))

    df = pd.DataFrame(
        {"PID": pids, "age": ages, "event": events, "timestamp": timestamps}
    )
    return df
