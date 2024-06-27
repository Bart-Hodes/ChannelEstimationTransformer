import os
import json
from tabulate import tabulate


def flatten_list(nested_list):
    """Flatten a nested list."""
    flat_list = []
    for item in nested_list:
        if isinstance(item, list):
            flat_list.extend(flatten_list(item))
        else:
            flat_list.append(item)
    return flat_list


def find_min_max_difference_per_file(directory):
    report = []

    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            filepath = os.path.join(directory, filename)
            try:
                with open(filepath, "r") as file:
                    data = json.load(file)
                    if isinstance(data, list):
                        flat_numbers = flatten_list(data)
                        numbers = [
                            value
                            for value in flat_numbers
                            if isinstance(value, (int, float))
                        ]
                        if numbers:
                            min_value = min(numbers)
                            max_value = max(numbers)
                            difference = max_value - min_value
                            report.append([filename, min_value, max_value, difference])
                        else:
                            report.append([filename, "No numerical data found", 0, 0])
                    else:
                        report.append([filename, "JSON content is not a list", 0, 0])
            except json.JSONDecodeError:
                report.append([filename, "Error decoding JSON", 0, 0])
            except Exception as e:
                report.append([filename, f"An error occurred: {e}", 0, 0])

    # Sort the report by filename
    report.sort(key=lambda x: x[3])
    return report


# Example usage:
directory_path = "weight_export"
report = find_min_max_difference_per_file(directory_path)

# Print the report as a table
headers = ["Filename", "Min Value", "Max Value", "Difference"]
print(tabulate(report, headers=headers, tablefmt="pretty"))
