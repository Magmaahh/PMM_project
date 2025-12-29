from config import *
from utils import prepare_data
from functions import *

if __name__ == "__main__":
    # Prepare train and test data
    prefix_training_log, prefix_testing_log, test_set, activities = prepare_data(TRAIN_DATA_PATH, TEST_DATA_PATH, PREFIX_LENGTH)

    X_train = [trace["encoding"] for trace in prefix_training_log]
    y_train = [trace["label"] for trace in prefix_training_log]

    X_test = [trace["encoding"] for trace in prefix_testing_log]
    y_test = [trace["label"] for trace in prefix_testing_log]

    # Optimize and train the model
    model = optimize_and_train_model(X_train, y_train, params)

    # Test the model
    test_model(model, X_test, y_test)

    # Convert the trained tree to code
    tree_to_code(model, activities)

    # Extract and evaluate recommendations
    recommendations = extract_recommendations(tree=model, feature_names=activities, class_values=model.classes_, prefix_set=prefix_testing_log)
    evaluate_recommendations(test_set, recommendations)