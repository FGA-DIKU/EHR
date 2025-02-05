SPECIAL_TOKEN_ABSPOS_ADJUSTMENT = (
    1e-3  # used for cls and sep token to ensure correct ordering
)

### Special tokens ###
UNKNOWN_TOKEN = "[UNK]"
CLS_TOKEN = "[CLS]"
SEP_TOKEN = "[SEP]"
MASK_TOKEN = "[MASK]"
PAD_TOKEN = "[PAD]"

### Columns ###
PID_COL = "PID"
TIMESTAMP_COL = "TIMESTAMP"
BIRTHDATE_COL = "BIRTHDATE"
DEATHDATE_COL = "DEATHDATE"
ABSPOS_COL = "abspos"

### Schema ###
SCHEMA = {
    "PID": "str",
    "age": "float32",
    "abspos": "float64",
    "segment": "int32",
}

FEATURES_SCHEMA = {**SCHEMA, "concept": "str"}
TOKENIZED_SCHEMA = {**SCHEMA, "concept": "int32"}
