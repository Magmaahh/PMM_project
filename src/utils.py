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
def encode_traces(traces, activities, prefix_length):
    activity_index = {activity: idx for idx, activity in enumerate(activities)}
    num_activities = len(activities)

    for trace in traces:
        encoding = [0] * num_activities

        for i, activity in enumerate(trace["activities"]):
            if i >= prefix_length:
                break
            if activity in activity_index:
                encoding[activity_index[activity]] = 1

        trace["encoding"] = encoding

    return traces

# Loads raw traces from an XES file;
# returns the list of traces in the file and the sorted list of unique activities
def load_raw_traces(xes_path):
    log = xes_importer.apply(xes_path)

    traces = []
    activity_set = set()

    for trace in log:
        case_id = trace.attributes.get("concept:name")
        label = trace.attributes.get("label")

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

    return traces, sorted(activity_set)

# Prepares training and testing data from raw XES files;
# saves the processed data to CSV files and returns the processed traces
def prepare_data(raw_training_data, raw_test_data, prefix_length):
    # Load and process training data
    traces, activities = load_raw_traces(raw_training_data)
    training_traces = encode_traces(traces, activities, prefix_length)  

    # Load and process testing data
    traces, _ = load_raw_traces(raw_test_data)
    testing_traces = encode_traces(traces, activities, prefix_length)

    # Save processed data to CSV files
    data_path = f"data/processed/{prefix_length}_prefix/"
    os.makedirs(data_path, exist_ok=True)

    save_traces(training_traces, activities, os.path.join(data_path, "training_data.csv"))
    save_traces(testing_traces, activities, os.path.join(data_path, "testing_data.csv"))

    return training_traces, testing_traces