import re
import matplotlib.pyplot as plt
import numpy as np

import random


def parse_log_file(log_file):
    results = {}

    with open(log_file, "r") as file:
        lines = file.readlines()

        # Regular expression patterns for matching relevant lines
        model_pattern = re.compile(r"^Model_name: (.+)")
        loss_pattern = re.compile(
            r"\| end of epoch\s+(\d+)\s+\|\s+.*\|\s+valid loss\s+(.+)\s+\|\s+Loss pred len\s+(.+)"
        )

        current_model = None
        last_loss_pred_len = None

        for line in lines:
            # Match model name
            model_match = model_pattern.match(line)
            if model_match:
                current_model = model_match.group(1)
                results[current_model] = []
                last_loss_pred_len = None
                continue

            # Match loss line
            loss_match = loss_pattern.match(line)
            if loss_match:
                epoch, _, loss_pred_len_str = loss_match.groups()
                numbers = [
                    float(s) + 0.005
                    for s in loss_pred_len_str.split()
                    if s.replace(".", "", 1).isdigit()
                ]
                results[current_model] = numbers

    return results


log_file = "label_length.txt"
results = parse_log_file(log_file)
print(results)
# Plotting
plt.figure(figsize=(10, 6))
for model, loss_pred_len_list in results.items():
    plt.plot(10 * np.log(loss_pred_len_list), label=model)

plt.xlabel("Epoch")
plt.ylabel("Loss NMSE")
plt.title("Loss vs label length SNR:40")
plt.legend(["label_len:5", "label_len:10", "label_len:15", "label_len:20"])
plt.grid(True)
plt.savefig("loss_VS_label_len.png")
