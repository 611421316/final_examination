import json
import os
import random
import argparse

def create_sample(input_path, output_task_dir, output_gt_dir, n_samples):
    os.makedirs(output_task_dir, exist_ok=True)
    os.makedirs(output_gt_dir, exist_ok=True)

    records = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    
    # ensure stability for reproduction
    random.seed(42)
    sampled = random.sample(records, min(n_samples, len(records)))
    
    for i, rec in enumerate(sampled, start=1):
        task_data = {
            "type": "user_behavior_simulation",
            "user_id": rec["user_id"],
            "item_id": rec["item_id"]
        }
        gt_data = {
            "stars": rec["stars"],
            "review": rec["text"]
        }
        
        with open(os.path.join(output_task_dir, f"task_{i}.json"), 'w', encoding='utf-8') as f:
            json.dump(task_data, f, indent=2)
            
        with open(os.path.join(output_gt_dir, f"groundtruth_{i}.json"), 'w', encoding='utf-8') as f:
            json.dump(gt_data, f, indent=2)
            
    print(f"Successfully generated {len(sampled)} sample tasks in {output_task_dir} and {output_gt_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/test_review_subset.json")
    parser.add_argument("--task-dir", default="dummy_tasks")
    parser.add_argument("--gt-dir", default="dummy_groundtruth")
    parser.add_argument("--n", type=int, default=5)
    args = parser.parse_args()
    
    create_sample(args.input, args.task_dir, args.gt_dir, args.n)
