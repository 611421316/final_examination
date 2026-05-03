import sys
import os
import json
import re
from typing import Any
from first_crew.crew import FirstCrew


# Make src importable
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))


def extract_prediction(output: Any) -> dict:
    """
    Extract {"stars": number, "review": string} from CrewAI output.
    """
    raw = getattr(output, "raw", output)

    if not isinstance(raw, str):
        raw = str(raw)

    text = raw.strip()

    # Remove markdown JSON block if exists
    text = re.sub(r"^```json\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    # Extract first JSON object
    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end != -1 and end > start:
        text = text[start:end + 1]

    try:
        data = json.loads(text)

        if "predicted_stars" in data and "stars" not in data:
            data["stars"] = data["predicted_stars"]

        return {
            "stars": float(data.get("stars", 3)),
            "review": data.get("review", "")
        }

    except Exception:
        return {
            "stars": 3.0,
            "review": ""
        }


def run_evaluation():
    test_path = "data/test_review_subset.json"

    with open(test_path, "r", encoding="utf-8") as f:
        test_data = [json.loads(line) for line in f if line.strip()]

    test_data = test_data[:5]

    predictions = []
    ground_truths = []

    for index, sample in enumerate(test_data, start=1):
        user_id = sample["user_id"]
        item_id = sample["item_id"]
        true_rating = float(sample["stars"])  # or sample["rating"] if your key is rating

        print(f"\n===== Test sample {index} =====")
        print(f"user_id: {user_id}")
        print(f"item_id: {item_id}")
        print(f"true rating: {true_rating}")

        inputs = {
            "user_id": user_id,
            "item_id": item_id,
            "query": f"Predict rating for user_id {user_id} and item_id {item_id}"
        }

        result = FirstCrew().sequential_crew().kickoff(inputs=inputs)

        parsed = extract_prediction(result)
        predicted_rating = float(parsed["stars"])

        print(f"predicted rating: {predicted_rating}")
        print(f"generated review: {parsed['review']}")

        predictions.append(predicted_rating)
        ground_truths.append(true_rating)

    return predictions, ground_truths


def calculate_metrics(predictions, ground_truths):
    n = len(predictions)

    exact_correct = 0
    total_absolute_error = 0

    for pred, true in zip(predictions, ground_truths):
        if pred == true:
            exact_correct += 1

        total_absolute_error += abs(pred - true)

    accuracy = exact_correct / n
    mae = total_absolute_error / n

    print("\n========== FINAL EVALUATION ==========")
    print(f"Total test samples: {n}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"MAE: {mae:.4f}")

    return accuracy, mae



if __name__ == "__main__":
    preds, gts = run_evaluation()
    calculate_metrics(preds, gts)