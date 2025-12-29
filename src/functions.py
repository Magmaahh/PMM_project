from sklearn.metrics import accuracy_score, f1_score
from sklearn.tree import _tree

# Trains the model, makes predictions, and evaluates performance
def train_test_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')

    print(f"Model Accuracy: {accuracy:.4f}")
    print(f"Model F1 Score: {f1:.4f}")

    return model

''' function to convert a decision tree to code
def tree_to_code(tree, feature_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    print("def tree({}):".format(", ".join(feature_names)))

    def recurse(node, depth):
        indent = "  " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]

            print ("{}if {} NOT done:".format(indent, name))
            recurse(tree_.children_left[node], depth + 1)
            print ("{}else:  # if {} done".format(indent, name))
            recurse(tree_.children_right[node], depth + 1)
        else:
            print ("{}return {}".format(indent, tree_.value[node]))

    recurse(0, 1)
'''

# Extracts all paths leading to positive classification
def extract_positive_paths(tree, feature_names):
    tree_ = tree.tree_
    paths = []

    def recurse(node, path):
        if tree_.feature[node] == _tree.TREE_UNDEFINED:
            values = tree_.value[node][0]
            confidence = values[0]
            if confidence > (sum(values) / 2):
                print(f"Path: {path}, confidence: {confidence}")
                paths.append({
                    "conditions": path.copy(),
                    "confidence": confidence
                })
            return

        activity = feature_names[tree_.feature[node]]

        recurse(tree_.children_left[node],  path + [(activity, 0)])
        recurse(tree_.children_right[node], path + [(activity, 1)])

    recurse(0, [])
    return paths

# Extracts recommendations for each trace in the prefix set
def extract_recommendations(tree, feature_names, class_values, prefix_set):
    activity_index = {a: i for i, a in enumerate(feature_names)}
    positive_paths = extract_positive_paths(tree, feature_names)
    recommendations = []

    for trace in prefix_set:
        encoding = trace["encoding"]
        predicted_label = tree.predict([encoding])[0]

        # Positive prediction: no recommendation
        if predicted_label == class_values[0]:
            recommendations.append({"trace": trace["case_id"], "recommendation": None})
            continue

        # Negative prediction: find best compatible positive path, if any
        compatible_paths = []
        for path in positive_paths:
            compatible = True
            for activity, presence in path["conditions"]:
                idx = activity_index[activity]

                # prefix contradicts the path
                if encoding[idx] == 1 and presence == 0:
                    compatible = False
                    break

            if compatible:
                compatible_paths.append(path)

        # No compatible positive path
        if not compatible_paths:
            recommendations.append({"trace": trace["case_id"], "recommendation": set()})
            continue

        # Select highest-confidence path
        best_path = max(compatible_paths, key=lambda p: p["confidence"])

        # Extract recommendations
        recs = set()
        for activity, presence in best_path["conditions"]:
            idx = activity_index[activity]
            if encoding[idx] != presence:
                if presence == 1:
                    recs.add(f"{activity} has to be executed")
                else:
                    recs.add(f"{activity} does not have to be executed")

        recommendations.append({"trace": trace["case_id"], "recommendation": recs})

    return recommendations

def evaluate_reccomendations(test_set, reccomendations):
    for trace in test_set:
        pass