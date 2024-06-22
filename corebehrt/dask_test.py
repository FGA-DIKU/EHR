import dask.dataframe as dd
from corebehrt.classes.tokenizer import EHRTokenizer
from corebehrt.functional.split import split_pids_into_pt_ft_test
from corebehrt.functional.convert import convert_to_sequences


df = dd.read_csv('../outputs/features/features/features.csv')
pids = df.PID.unique().compute().tolist()

pt_pids, ft_pids, test_pids = split_pids_into_pt_ft_test(pids, 0.8, 0.1, 0.1)

tokenizer = EHRTokenizer()
df_pt = df[df.PID.isin(pt_pids)]
df_pt = tokenizer(df_pt)
tokenizer.freeze_vocabulary()
df_ft_test = df[df.PID.isin(set(ft_pids+test_pids))]
df_ft_test = tokenizer(df_ft_test)
df_test = df_ft_test[df_ft_test.PID.isin(test_pids)]

# df_pt = df_pt.compute()
features_test, pids_test = convert_to_sequences(df_test)
print(features_test)
print(df_pt.head(50))





#print(df[result.PID=='01ff265a-fbe6-317f-3157-f97c404f4cf5'].head(20))
#print(df[df.PID=='01ff265a-fbe6-317f-3157-f97c404f4cf5'].compute())

#df = pd.read_csv('../outputs/features/features/features.csv')
#print(df.head(40))

