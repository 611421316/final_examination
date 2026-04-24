#!/usr/bin/env python
import sys
import warnings
import json

from first_crew.crew import FirstCrew

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

import re

def extract_json_from_output(raw_output: str) -> dict:
    """Extract and sanitize JSON from LLM raw output."""
    text = str(raw_output).strip()
    
    # Fix double curly braces {{ }} -> { }
    text = text.replace('{{', '{').replace('}}', '}')
    
    # Try to find JSON object pattern in the text
    match = re.search(r'\{[^{}]*"stars"[^{}]*"review"[^{}]*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    
    # Fallback: try parsing the entire text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Last resort: return raw output wrapped in a dict
        return {"stars": None, "review": text, "_parse_error": True}


def run():
    """
    Run the crew using the first testing sample.
    """
    # === Step 1: Understand the Target & Dataset ===
    # Read the first entry from the test set as a demo
    test_json_path = "data/test_review_subset.json"
    with open(test_json_path, 'r', encoding='utf-8') as f:
        first_line = f.readline()
        test_case = json.loads(first_line)

    inputs = {
        'user_id': test_case['user_id'],
        'item_id': test_case['item_id']
    }

    print(f"Starting Prediction for User: {inputs['user_id']} | Item: {inputs['item_id']}")

    # === Step 7: Execute the Prediction ===
    try:
        result = FirstCrew().crew().kickoff(inputs=inputs)
        
        # Parse and sanitize the LLM output into clean JSON
        report = extract_json_from_output(result.raw)
        
        # Write clean report
        with open('report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\n=== Prediction Completed ===")
        print(f"Stars: {report.get('stars')}")
        print(f"Review: {report.get('review', '')[:100]}...")
        print(f"Results written to report.json")
        
    except Exception as e:
        raise Exception(f"An error occurred while running the crew: {e}")

def train():
    pass

def replay():
    pass

def test():
    pass

def run_with_trigger():
    pass
