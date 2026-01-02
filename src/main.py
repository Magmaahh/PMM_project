from config import *
from utils import prepare_data
from functions import *

if __name__ == "__main__":
    test_results = {}
    eval_results = {}

    print(f"=== STARTING SESSION ===")

    for PREFIX_LENGTH in PREFIX_LENGTH:
        print(f"\n\n=== Processing with Prefix Length: {PREFIX_LENGTH} ===\n")

        # Prepare train and test data
        prefix_training_log, prefix_testing_log, test_set, activities = prepare_data(TRAIN_DATA_PATH, TEST_DATA_PATH, PREFIX_LENGTH)

        X_train = [trace["encoding"] for trace in prefix_training_log]
        y_train = [trace["label"] for trace in prefix_training_log]

        X_test = [trace["encoding"] for trace in prefix_testing_log]
        y_test = [trace["label"] for trace in prefix_testing_log]

        # Optimize and train the model
        model = optimize_and_train_model(X_train, y_train, params)

        # Test the model
        test_results[PREFIX_LENGTH] = test_model(model, X_test, y_test)

        # Convert the trained model to code, display its structure and save its plot
        show_and_save_tree(model, activities, TREES_PLOTS_PATH, PREFIX_LENGTH)

        # Extract and evaluate recommendations
        recommendations = extract_recommendations(tree=model, feature_names=activities, class_values=model.classes_, prefix_set=prefix_testing_log)
        eval_results[PREFIX_LENGTH] = evaluate_recommendations(test_set, recommendations)

    print(f"\n\n=== Summary of Results ===\n")
    for prefix_len, metrics in eval_results.items():
        print(f"Prefix Length {prefix_len}:")
        print(f"Model testing: Accuracy={test_results[prefix_len]['accuracy']:.4f}, Precision={test_results[prefix_len]['precision']:.4f}, Recall={test_results[prefix_len]['recall']:.4f}, F1-Score={test_results[prefix_len]['f_measure']:.4f}")
        print(f"Recommendations evaluation: Accuracy={metrics['accuracy']:.4f}, Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}, F1-Score={metrics['f1_score']:.4f}\n")