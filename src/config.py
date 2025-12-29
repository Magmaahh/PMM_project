# File paths for raw train/test datasets
TRAIN_DATA_PATH = "data/raw/Production_avg_dur_training_0-80.xes"
TEST_DATA_PATH = "data/raw/Production_avg_dur_testing_80-100.xes"

# Prefix length used for data processing
PREFIX_LENGTH = 5

# Default parameters for the decision tree classifier
params = {
    "max_depth": 3,
    "min_samples_split": 5,
    "min_samples_leaf": 2
}