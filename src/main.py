from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt

from config import *
from utils import prepare_data
from functions import *

if __name__ == "__main__":
    # Prepare train and test data
    train_set, test_set, activities = prepare_data(TRAIN_DATA_PATH, TEST_DATA_PATH, PREFIX_LENGTH)

    X_train = [trace["encoding"] for trace in train_set]
    y_train = [trace["label"] for trace in train_set]

    X_test = [trace["encoding"] for trace in test_set]
    y_test = [trace["label"] for trace in test_set]

    # Initialize the model
    model = DecisionTreeClassifier(
        max_depth=params["max_depth"],
        min_samples_split=params["min_samples_split"],
        min_samples_leaf=params["min_samples_leaf"],
        random_state=42
    )

    # Train and test the model
    model = train_test_model(model, X_train, y_train, X_test, y_test)

    # Extract and evaluate recommendations
    recommendations = extract_recommendations(tree=model, feature_names=activities, class_values=["true","false"], prefix_set=test_set)

    for rec in recommendations:
        print(rec)

    evaluate_reccomendations(test_set, recommendations)