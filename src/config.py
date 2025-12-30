# File paths for raw train/test datasets
TRAIN_DATA_PATH = "data/raw/Production_avg_dur_training_0-80.xes"
TEST_DATA_PATH = "data/raw/Production_avg_dur_testing_80-100.xes"

# Prefix length used for data processing
PREFIX_LENGTH = [5, 10]

# Grid search parameters for Decision Tree Classifier
params = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [3, 5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4, 5]
    }