# from dask import dataframe as dd
# from corebehrt.functional.features.normalize import min_max_normalize


# class ValuesNormalizer:
#     """
#     A class to load normalise values in data frames.
#     Expects a 'result' column and 'concept' column to be present.
#     """

#     @staticmethod
#     def min_max_normalize_results(concepts: dd.DataFrame, num_bins=100, normalize_values=False) -> dd.Series:
#         concepts = concepts.shuffle(
#             "CONCEPT"
#         )  # Shuffle to ensure that the same concept is in the same partition
#         #concepts = concepts.reset_index(drop=True)

#         # Has to be be assigned inside here due to some weird dask behaviour
#         concepts["RESULT"] = concepts.map_partitions(
#             lambda partition: partition.groupby("CONCEPT")["RESULT"].transform(
#                 lambda series: ValuesNormalizer.min_max_normalize(
#                     series, num_bins=num_bins, normalize_values=normalize_values
#                 )
#             )
#         )

#         # Add index + order
#         concepts["index"] = concepts.index
#         values = concepts.dropna(subset=["RESULT"])
#         values["CONCEPT"] = values["RESULT"]
#         concepts["order"] = 0
#         values["order"] = 1
#         concatted = dd.concat([concepts, values])
#         return concatted.drop(columns=["RESULT"], axis=1)

#     @staticmethod
#     def min_max_normalize(series: dd.Series, num_bins=100, normalize_values=False) -> dd.Series:
#         """
#         Normalises the a column of the given Series using min-max normalisation and converts to 100 VAL_{i} bins
#         """
#         normalized_values = min_max_normalize(series) if normalize_values else series
#         normalized_values = normalized_values.dropna()
#         val_mask = normalized_values != "UNIQUE"
#         normalized_values = (
#             normalized_values.where(
#                 ~val_mask, normalized_values[val_mask].mul(num_bins)
#             )
#             .astype(str)
#             .str.split(".")
#             .str[0]
#         )  # using .astype(int) for float->int conversion doesnt work properly
#         return "VAL_" + normalized_values