from sklearn import tree
from sklearn.tree import _tree, DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from matplotlib import pyplot as plot
import os
import random

# Optimizes and trains the Decision Tree model using Grid Search
def optimize_and_train_model(X_train, y_train, params):
    print(f"\n--- Hyperparameter Optimization via Grid Search ---\n")

    print(f"Optimizing with Grid Search...")
    model = DecisionTreeClassifier(random_state=42)
    grid_search = GridSearchCV(
        model, 
        param_grid=params, 
        cv=5, 
        scoring='f1_macro', 
        n_jobs=-1
    )

    print("\nTraining the model with best found hyperparameters...\n")
    grid_search.fit(X_train, y_train)
    print(f"Best Hyperparameters: {grid_search.best_params_}")

    return grid_search.best_estimator_

# Tests the trained model and outputs performance metrics
def test_model(model, X_test, y_test):
    print(f"\n\n--- Model Testing ---\n")

    print("Testing model...")
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f_measure = f1_score(y_test, y_pred, average='binary', pos_label='true')
    precision = precision_score(y_test, y_pred, average='binary', pos_label='true')
    recall = recall_score(y_test, y_pred, average='binary', pos_label='true')
    
    print("\nTest Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F-measure: {f_measure:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}\n")

    return {
        "accuracy": accuracy,
        "f_measure": f_measure,
        "precision": precision,
        "recall": recall
    }

# Converts the decision tree into a readable code format, displays its structure, and saves its plot
def show_and_save_tree(model, feature_names, plots_path, prefix_length):
    tree_ = model.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    print("\n\n--- Decision Tree Structure ---\n")
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

    plot.figure(figsize=(30,20))
    tree.plot_tree(model, filled=True, rounded=True)
    os.makedirs(plots_path, exist_ok=True)
    plot.savefig(os.path.join(plots_path, f"tree_prefix{prefix_length}.png"))

# Extracts positive paths from the decision tree
def extract_positive_paths(tree, feature_names):
    tree_ = tree.tree_
    paths = []

    print(f"\n\n--- Extracting Positive Paths from the Decision Tree ---")

    # Identify index of positive class
    classes = list(tree.classes_)
    if "true" not in classes:
        raise ValueError("Positive class 'true' not found in model classes")
    pos_idx = classes.index("true")

    def recurse(node, path):
        if tree_.feature[node] == _tree.TREE_UNDEFINED:
            values = tree_.value[node][0]
            total = values.sum()
            confidence = values[pos_idx] / total if total > 0 else 0

            # Keep all leaves predicting positive class
            if values[pos_idx] == max(values):
                paths.append({
                    "conditions": path.copy(),
                    "confidence": confidence
                })
            return

        activity = feature_names[tree_.feature[node]]
        recurse(tree_.children_left[node],  path + [(activity, 0)])
        recurse(tree_.children_right[node], path + [(activity, 1)])

    recurse(0, [])

    print(f"\nExtracted {len(paths)} positive paths from the decision tree:")
    for i, p in enumerate(paths):
        print(f"\nPath {i+1}:")
        for act, pres in p["conditions"]:
            print(f"  - {act}: {'DONE' if pres == 1 else 'NOT DONE'}")
        print(f"  Confidence: {p['confidence']:.4f}")

    return paths

# Extracts recommendations for each prefix in the prefix set
def extract_recommendations(tree, feature_names, class_values, prefix_set):
    activity_index = {a: i for i, a in enumerate(feature_names)}
    positive_paths = extract_positive_paths(tree, feature_names)

    # Identify positive class label
    if "true" not in class_values:
        raise ValueError("Positive class 'true' not found in model classes")
    pos_label = class_values[list(class_values).index("true")]

    print("\n\n=== Recommendation Extraction ===")
    recommendations = []
    for trace in prefix_set:
        case_id = trace["case_id"]
        encoding = trace["encoding"]
        predicted_label = tree.predict([encoding])[0]
        
        print(f"\n--- Trace {case_id} ---")
        print(f"Predicted label: {predicted_label}")
        print("\nActivities in the prefix:")
        for i, v in enumerate(encoding):
            if v == 1:
                print(f"  • {feature_names[i]}")

        # If already predicted positive, no recommendation
        if str(predicted_label).lower() == pos_label:
            print("\nAlready predicted as positive, no recommendation needed")
            print("--------------------------------")
            recommendations.append({
                "trace": case_id,
                "recommendation": None
            })
            continue

        # If negative predicted, find compatible positive paths
        compatible_paths = []
        for path in positive_paths:
            compatible = True
            for activity, presence in path["conditions"]:
                idx = activity_index[activity]

                # Contradiction: activity already executed, but path requires it to never happen
                if encoding[idx] == 1 and presence == 0:
                    compatible = False
                    break

            if compatible:
                compatible_paths.append(path)

        # If no compatible positive paths, empty set of recommendations
        if not compatible_paths:
            print("\nNo compatible positive paths found")
            print("--------------------------------")
            recommendations.append({
                "trace": case_id,
                "recommendation": set()
            })
            continue

        # Compatible paths found: select most confident path (with random tie-break)
        print(f"\nFound {len(compatible_paths)} compatible positive paths.")
        print("Selecting the most confident path...")
        max_conf = max(p["confidence"] for p in compatible_paths)
        best_paths = [p for p in compatible_paths if p["confidence"] == max_conf]
        best_path = random.choice(best_paths)

        print(f"\nSelected most confident positive path (confidence = {best_path['confidence']:.4f}):")
        for act, pres in best_path["conditions"]:
            print(f"  - {act}: {'DONE' if pres == 1 else 'NOT DONE'}")
        print("--------------------------------")

        # Generate recommendations based on the best path
        rec = set()
        for activity, presence in best_path["conditions"]:
            idx = activity_index[activity]

            if presence == 1 and encoding[idx] == 1:
                continue  # already satisfied

            verdict = "has to be executed" if presence == 1 else "does not have to be executed"
            rec.add((activity, verdict))

        recommendations.append({
            "trace": case_id,
            "recommendation": rec
        })

    return recommendations

# Evaluates the recommendations against the full test set
def evaluate_recommendations(test_set, recommendations):
    traces_map = {t["case_id"]: t for t in test_set}
    TP = FP = TN = FN = 0

    print("\n\n--- Recommendations evaluation for negative predicted traces: ---")
    for rec in recommendations:
        # Skip traces with no recommendations (= positive predicted traces)
        if rec["recommendation"] is None: 
            continue

        trace = traces_map.get(rec["trace"])
        actual_activities = set(trace["activities"])
        ground_truth_pos = (str(trace["label"]).lower() == "true")
        rec_set = rec["recommendation"]

        print(f"\nTrace ID: {rec['trace']}")
        print(f"Label (true = fast/false = slow): {ground_truth_pos}")

        print(f"\nRecommendations:")
        if not len(rec_set) == 0:
            for activity, verdict in rec_set:
                print(f"  - {activity}: {verdict}")
        else:
            print("No recommendations could be made")

        print("\nExecuted activities:")
        for act in actual_activities:
            print(f"  • {act}")
        print("--------------------------------")

        if not rec_set:
            followed = False
        else:
            followed = True
            for activity, verdict in rec_set:
                if verdict == "has to be executed" and activity not in actual_activities:
                    followed = False; break
                if verdict == "does not have to be executed" and activity in actual_activities:
                    followed = False; break

        if followed and ground_truth_pos: TP += 1
        elif not followed and not ground_truth_pos: TN += 1
        elif followed and not ground_truth_pos: FP += 1
        elif not followed and ground_truth_pos: FN += 1

    # Metrics computation
    print("\nEvaluating recommendations and computing metrics...")
    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
    prec = TP / (TP + FP) if (TP + FP) > 0 else 0
    rec_val = TP / (TP + FN) if (TP + FN) > 0 else 0
    f_score = (2 * prec * rec_val / (prec + rec_val)) if (prec + rec_val) > 0 else 0

    print("\nRecommendations evaluation results:")
    print(f"Classification: TP={TP}, FP={FP}, TN={TN}, FN={FN}")
    print(f"Accuracy_recc:  {accuracy:.4f}")
    print(f"Precision_recc: {prec:.4f}")
    print(f"Recall_recc:    {rec_val:.4f}")
    print(f"F-measure_recc: {f_score:.4f}")

    return {
        "accuracy": accuracy,
        "precision": prec,
        "recall": rec_val,
        "f1_score": f_score
    }