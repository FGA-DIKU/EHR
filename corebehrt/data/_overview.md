### batch.py
Should be simplified

BatchTokenize shouldn't be needed if we convert to Dataframe Tokenization
-> we can reuse conceptloader if the features are stored in a dataframe

Hard to judge how much is needed

### concept_loader.py
ConceptLoader - convert everything to functional

### creators.py
Probably fine as is

### dataset.py
Probably fine as is

### featuremaker.py
FeatureMaker (if Dataframe format) should just return dataframe and not create features 
 - Will require minor revamp

### filter.py
We need to decide whether it's still dataframe format at this point then we can simplify
CodeTypeFilter
 - _filter_patient should work on patient, not need Data, can be simplified
 - Is CodeTypeFilter even needed? We select code type when creatin features

PatientFilter: Use functional
 - Is nested functionals an issue?
 - Move select entries to functional
 - Move exclude pids to functional

### mask.py
Simple ConceptMasker and functional call
Can't we just use transformers implementation?

### prepare_data.py
Pipeline component, save for later ->corebehrt.pipeline

### split.py
Simple functional

### tokenizer.py
Can be simplified with Dataframe format
 - How is SEP tokens created then? (Solved)
Either with
`sep_df = df[df['segment'] != df['segment'].shift(-1)].copy()` or `sep_df = df.groupby(['person_id', 'segment']).last().reset_index()`

### utils.py
Convert to functional->simply moving functions

