import json
with open("/root/ShareGPT_V3_unfiltered_cleaned_split.json") as f:
    dataset = json.load(f)
# Filter out the conversations with less than 2 turns.
dataset = [
    data for data in dataset
    if len(data["conversations"]) >= 2
]
new_dataset = []
# Only keep the first two turns of each conversation.
for data in dataset:
    new_data = data
    new_data["conversations"] = [data["conversations"][0], data["conversations"][1]]
    new_dataset.append(new_data)
with open("/root/dataset.json", "w") as f:
    f.write(json.dumps(new_dataset[:1500]))