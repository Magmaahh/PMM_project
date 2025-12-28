from sklearn.metrics import accuracy_score, f1_score

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

# ToDo

def extract_reccomendations(tree, feature_names, class_values, prefix_set):
    pass

def evaluate_reccomendations(test_set, reccomendations):
    pass