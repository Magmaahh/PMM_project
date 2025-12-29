from sklearn.metrics import accuracy_score, f1_score
from sklearn.tree import _tree, DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
import random

# Optimizes and trains the Decision Tree model using Grid Search
def optimize_and_train_model(X_train, y_train, params):
    model = DecisionTreeClassifier(random_state=42)
    grid_search = GridSearchCV(
        model, 
        param_grid=params, 
        cv=5, 
        scoring='f1_macro', 
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    print(f"Best Hyperparameters: {grid_search.best_params_}")
    return grid_search.best_estimator_

# Tests the trained model and outputs accuracy and F-measure
def test_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f_measure = f1_score(y_test, y_pred, average='binary', pos_label='true')

    print(f"\n--- Model Prediction Performance ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F-measure (Outcome): {f_measure:.4f}")

# Converts the decision tree into a readable code format
def tree_to_code(tree, feature_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    print("--- Decision Tree Structure ---")
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

# Extracts positive paths from the decision tree
def extract_positive_paths(tree, feature_names, class_values):
    tree_ = tree.tree_
    paths = []

    # Identify index of the positive class
    class_list = list(class_values)
    if "true" not in class_list:
        print("Warning: 'true' label not found in model classes!")
        return []
    pos_idx = class_list.index("true")

    # Recursive function to traverse the tree
    def recurse(node, path):
        # Leaf node
        if tree_.feature[node] == _tree.TREE_UNDEFINED:
            values = tree_.value[node][0]
            total = values.sum()
            confidence = values[pos_idx] / total if total > 0 else 0

            if confidence > 0.5:
                paths.append({
                    "conditions": path.copy(),
                    "confidence": confidence
                })
            return
        
        # Internal node
        activity = feature_names[tree_.feature[node]]
        recurse(tree_.children_left[node],  path + [(activity, 0)])
        recurse(tree_.children_right[node], path + [(activity, 1)])

    recurse(0, [])

    return paths

# Extracts recommendations for each trace in the prefix set
def extract_recommendations(tree, feature_names, class_values, prefix_set):
    activity_index = {a: i for i, a in enumerate(feature_names)}
    positive_paths = extract_positive_paths(tree, feature_names, class_values)
    
    recommendations = []
    for trace in prefix_set:
        encoding = trace["encoding"]
        predicted_label = tree.predict([encoding])[0]

        # Requirement: skip if predicted positive
        if str(predicted_label).lower() == class_values[0].lower():
            recommendations.append({"trace": trace["case_id"], "recommendation": None})
            continue

        # Filter paths compliant with the current trace prefix
        compatible_paths = []
        for path in positive_paths:
            is_compatible = True
            for activity, presence in path["conditions"]:
                if encoding[activity_index[activity]] == 1 and presence == 0:
                    is_compatible = False
                    break
            if is_compatible:
                compatible_paths.append(path)

        if not compatible_paths:
            recommendations.append({"trace": trace["case_id"], "recommendation": set()})
            continue

        # Selection: randomly choose among paths with highest confidence (TO UPDATE)
        max_confidence = max(p["confidence"] for p in compatible_paths)
        best_paths = [p for p in compatible_paths if p["confidence"] == max_confidence]
        best_path = random.choice(best_paths)

        rec = set()
        for activity, presence in best_path["conditions"]:
            if encoding[activity_index[activity]] != presence:
                verdict = "execute" if presence == 1 else "skip"
                rec.add((activity, verdict))

        recommendations.append({"trace": trace["case_id"], "recommendation": rec})

    return recommendations

# Evaluates the recommendations against the full test set
def evaluate_recommendations(test_set, recommendations):
    traces_map = {t["case_id"]: t for t in test_set}
    TP = FP = TN = FN = 0

    print(f"\n--- Detailed Recommendation Evaluation ---")
    for rec in recommendations:
        if rec["recommendation"] is None: 
            continue

        trace = traces_map.get(rec["trace"])
        actual_activities = set(trace["activities"])
        ground_truth_pos = (str(trace["label"]).lower() == "true")
        rec_set = rec["recommendation"]

        print(f"\nTrace ID: {rec['trace']}")
        print(f"Actual Activities: {actual_activities}")
        print(f"Ground Truth Positive: {ground_truth_pos}")
        print(f"Recommendation Set: {rec_set}")

        if not rec_set:
            followed = False
        else:
            followed = True
            for activity, verdict in rec_set:
                if verdict == "execute" and activity not in actual_activities:
                    followed = False; break
                if verdict == "skip" and activity in actual_activities:
                    followed = False; break

        if followed and ground_truth_pos: TP += 1
        elif not followed and not ground_truth_pos: TN += 1
        elif followed and not ground_truth_pos: FP += 1
        elif not followed and ground_truth_pos: FN += 1

    # Metrics computation
    prec = TP / (TP + FP) if (TP + FP) > 0 else 0
    rec_val = TP / (TP + FN) if (TP + FN) > 0 else 0
    f_score = (2 * prec * rec_val / (prec + rec_val)) if (prec + rec_val) > 0 else 0

    print(f"Classification: TP={TP}, FP={FP}, TN={TN}, FN={FN}")
    print(f"Precision_recc: {prec:.4f}")
    print(f"Recall_recc:    {rec_val:.4f}")
    print(f"F-measure_recc: {f_score:.4f}")