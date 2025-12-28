from sklearn.tree import DecisionTreeClassifier

from config import *
from utils import prepare_data
from functions import train_test_model, extract_reccomendations, evaluate_reccomendations

if __name__ == "__main__":
    # Prepare train and test data
    train_set, test_set = prepare_data(TRAIN_DATA_PATH, TEST_DATA_PATH, PREFIX_LENGTH)

    X_train = [trace["encoding"] for trace in train_set]
    y_train = [trace["label"]for trace in train_set]

    X_test = [trace["encoding"] for trace in test_set]
    y_test = [trace["label"] == "true" for trace in test_set]

    # Initialize the model
    model = DecisionTreeClassifier(
        max_depth=params["max_depth"],
        min_samples_leaf=params["min_samples_leaf"],
        random_state=params["random_state"]
    )

    # Train and test the model
    train_test_model(model, X_train, y_train, X_test, y_test)

    # Extract and evaluate recommendations
    reccomendations = extract_reccomendations(model, feature_names=None, class_values=["true","false"], prefix_set=test_set)
    evaluate_reccomendations(test_set, reccomendations)