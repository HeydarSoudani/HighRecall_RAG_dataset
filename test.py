import json

# Input and output file paths
input_path = "c3_dataset_augmentation/mahta_code/outputs/results.jsonl"
output_path = "c3_dataset_augmentation/mahta_code/outputs/results_1.json"

# # Read the incorrect JSON file
# with open(input_path, "r", encoding="utf-8") as f:
#     data = json.load(f)

# print(len(data))

# # Write each JSON object to a new line
# with open(output_path, "w", encoding="utf-8") as f:
#     for item in data:
#         print(list(item.keys()))
#         f.write(json.dumps(item, ensure_ascii=False) + "\n")

# print(f"✅ Fixed JSONL written to {output_path}")





# Read the incorrect JSON file (list of JSONs)
with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)  # data is a list of dicts

# Convert list of dicts into a single dictionary
# Use an integer or string index as the key
data_dict = {str(i): item for i, item in enumerate(data)}

# Write to a JSON file in dictionary mode
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(data_dict, f, ensure_ascii=False, indent=2)

print(f"✅ Data written as a dictionary to {output_path}")


