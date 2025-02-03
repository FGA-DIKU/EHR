# Data Module Overview

## batch.py

- Should be simplified
- BatchTokenize shouldn't be needed if we convert to Dataframe Tokenization
- Hard to judge how much is needed

## concept_loader.py

- ConceptLoader - convert everything to functional
- ConceptLoaderLarge shouldn't be needed if we convert to Dataframe format

## creators.py

- Probably fine as is

## dataset.py

- Probably fine as is

## featuremaker.py

- FeatureMaker (if Dataframe format) should just return dataframe and not create features
- Will require minor revamp

## filter.py

- CodeTypeFilter:

  - _filter_patient should work on patient, not need Data, can be simplified

- PatientFilter:

- Use functional
- Is nested functionals an issue?

## mask.py

- Simple ConceptMasker and functional call

## prepare_data.py

- Pipeline component, save for later

## split.py

- Simple functional

## tokenizer.py

- Can be simplified with Dataframe format
- How is SEP tokens created then? (Solved)

Either with:

- `sep_df = df[df['segment'] != df['segment'].shift(-1)].copy()`
- `sep_df = df.groupby(['person_id', 'segment']).last().reset_index()`

## utils.py

- Convert to functional
