import json

user_id = "WnT9NIzQgLlILjPT0kEcsQ"
item_id = "nnwBdqGHIAJQ5QX9lHOtrQ"

def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

for path in ["data/filtered_user.json", "data/filtered_item.json", "data/test_review_subset.json"]:
    rows = load_jsonl(path)
    user_found = any(r.get("user_id") == user_id for r in rows)
    item_found = any(r.get("item_id") == item_id for r in rows)
    print(path, "user:", user_found, "item:", item_found)