SPECIAL_TOKEN_ABSPOS_ADJUSTMENT = (
    1e-3  # used for cls and sep token to ensure correct ordering
)
IGNORE_LOSS_INDEX = -100
### Special tokens ###
UNKNOWN_TOKEN = "[UNK]"
CLS_TOKEN = "[CLS]"
SEP_TOKEN = "[SEP]"
MASK_TOKEN = "[MASK]"
PAD_TOKEN = "[PAD]"
AGE_AT_CENSORING_TOKEN = "[AGE_AT_CENSORING]"

DEFAULT_VOCABULARY = {
    PAD_TOKEN: 0,
    CLS_TOKEN: 1,
    SEP_TOKEN: 2,
    UNKNOWN_TOKEN: 3,
    MASK_TOKEN: 4,
    AGE_AT_CENSORING_TOKEN: 5,
}

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

ADMISSION_ID_COL = "admission_id"
ADMISSION = "ADM_ADMISSION"
DISCHARGE = "ADM_DISCHARGE"

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
    "segment": "int32"
}

FEATURES_SCHEMA = {**SCHEMA, "code": "str"}
TOKENIZED_SCHEMA = {**SCHEMA, "code": "int32"}

### Folds and Splits###
TRAIN_KEY = "train"
VAL_KEY = "val"
TEST_KEY = "test"


CONCEPT_FEAT = "concept"
AGE_FEAT = "age"
ABSPOS_FEAT = "abspos"
SEGMENT_FEAT = "segment"
ATTENTION_MASK = "attention_mask"
TARGET = "target"

# combined outcomes
COMBINATIONS = "combinations"
TIMESTAMP_SOURCE = "timestamp_source"
WINDOW_HOURS_MIN = "window_hours_min"
WINDOW_HOURS_MAX = "window_hours_max"
PRIMARY = "primary"
SECONDARY = "secondary"
