from pm4py.objects.log.importer.xes import importer as xes_importer
import os
import csv

# Saves encoded traces to a CSV file with a header including case_id, label, and activities
def save_traces(traces, activities, file_path):
    header = ["case_id", "label"] + activities

    with open(file_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for trace in traces:
            row = ([trace["case_id"], trace["label"]] + trace["encoding"])
            writer.writerow(row)

# Encodes traces into binary vectors based on activity presence within a given prefix length
def encode_traces(traces, activities, prefix_length = None):
    activity_index = {activity: idx for idx, activity in enumerate(activities)}
    num_activities = len(activities)

    for trace in traces:
        encoding = [0] * num_activities

        for i, activity in enumerate(trace["activities"]):
            if prefix_length is not None and i >= prefix_length:
                break
            if activity in activity_index:
                encoding[activity_index[activity]] = 1
            else:
                encoding[activity_index["unknown"]] = 1

        trace["encoding"] = encoding

    print(f"Encoded {len(traces)} traces with prefix length {prefix_length if prefix_length is not None else 'full length'}\n")

    return traces

# Loads raw traces from an XES file;
# returns the list of traces in the file and the sorted list of unique activities
def load_raw_traces(xes_path):
    log = xes_importer.apply(xes_path)

    traces = []
    activity_set = set()

    for trace in log:
        case_id = trace.attributes.get("concept:name")
        label = str(trace.attributes.get("label")).lower()

        activities = []
        for event in trace:
            activity = event.get("concept:name")
            if activity is not None:
                activities.append(activity)
                activity_set.add(activity)

        if case_id is not None and label is not None:
            traces.append({
                "case_id": case_id,
                "label": label,
                "activities": activities
            })

    print(f"Loaded {len(traces)} traces from {xes_path}")
    print(f"Identified {len(activity_set)} unique activities (including \"unknown\")\n")

    sorted_activities = sorted(activity_set)
    sorted_activities.append("unknown")  # Add "unknown" activity for unseen activities during encoding

    return traces, sorted_activities

# Prepares training and testing data from raw XES files;
# saves the processed data to CSV files and returns the processed traces
def prepare_data(raw_training_data, raw_test_data, prefix_length):
    print(f"\n--- Data Preparation ---\n")

    # Load and process training data
    print(f"Preparing training data")
    raw_train_traces, activities = load_raw_traces(raw_training_data)
    train_traces = encode_traces(raw_train_traces, activities, prefix_length)  

    # Load and process testing data
    print(f"\nPreparing testing data")
    raw_test_traces, _ = load_raw_traces(raw_test_data)
    test_traces = encode_traces(raw_test_traces, activities, prefix_length)

    # Save processed data to CSV files
    data_path = "data/processed"
    prefix_data_path = os.path.join(data_path, f"{prefix_length}_prefix")
    os.makedirs(data_path, exist_ok=True)
    os.makedirs(prefix_data_path, exist_ok=True)

    save_traces(train_traces, activities, os.path.join(prefix_data_path, "training_data.csv"))
    save_traces(test_traces, activities, os.path.join(prefix_data_path, "testing_data.csv"))

    # Save full testing data without prefix limitation
    print(f"\nPreparing full testing data without prefix limitation")
    full_test_traces = encode_traces(raw_test_traces, activities, prefix_length=None)
    save_traces(full_test_traces, activities, os.path.join(data_path, "full_testing_data.csv"))

    return train_traces, test_traces, full_test_traces, activities