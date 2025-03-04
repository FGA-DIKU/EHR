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
PID_COL = "subject_id"
TIMESTAMP_COL = "time"
BIRTHDATE_COL = "birthdate"
DEATHDATE_COL = "deathdate"
ABSPOS_COL = "abspos"
CONCEPT_COL = "code"
AGE_COL = "age"
SEGMENT_COL = "segment"
VALUE_COL = "numeric_value"

# Codes
ADMISSION_CODE = "ADMISSION"
DISCHARGE_CODE = "DISCHARGE"
DEATH_CODE = "DOD"
BIRTH_CODE = "DOB"
BIRTHDATE_COL = "birthdate"
DEATHDATE_COL = "deathdate"

### Schema ###
SCHEMA = {
    "subject_id": "int64",
    "age": "float32",
    "abspos": "float64",
    "segment": "int32",
}

FEATURES_SCHEMA = {**SCHEMA, "code": "str"}
TOKENIZED_SCHEMA = {**SCHEMA, "code": "int32"}

### Folds and Splits###
TRAIN_KEY = "train"
VAL_KEY = "val"
TEST_KEY = "test"
