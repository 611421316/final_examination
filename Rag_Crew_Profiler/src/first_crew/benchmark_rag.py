import time
import os
import sys
from dotenv import load_dotenv

# Ensure environmental variables are loaded so CrewAI knows the correct DB location
load_dotenv()

from crewai_tools import JSONSearchTool
from langchain_community.embeddings import HuggingFaceEmbeddings

embedding_model = HuggingFaceEmbeddings(
    model_name='BAAI/bge-small-en-v1.5'
)
rag_config = {
    "embedding_model": {
        "provider": "sentence-transformer",
        "config": {
            "model_name": "BAAI/bge-small-en-v1.5"
        }
    }
}

# Connect exactly to the collections created by your run_id=1 indexing script
# OMIT json_path so CrewAI skips the synchronous chunking loops and mounts DB directly
filtered_user_rag_tool = JSONSearchTool(
    collection_name='benchmark_true_fresh_index_Filtered_User_1',
    config=rag_config
)

filtered_item_rag_tool = JSONSearchTool(
    collection_name='benchmark_true_fresh_index_Filtered_Item_1',
    config=rag_config
)

filtered_review_rag_tool = JSONSearchTool(
    collection_name='benchmark_true_fresh_index_Filtered_Review_1',
    config=rag_config
)

def run_benchmark():
    print("=== Starting Local RAG Benchmarking (Cached Indexes) ===")
    
    # Measure User RAG Tool retrieval time
    start_time = time.time()
    try:
        print("\nQuerying Filtered User RAG Tool...")
        res = filtered_user_rag_tool._run(search_query="What is the user's average stars and preferences?")
        user_time = time.time() - start_time
        print(f"User Tool Retrieval Time: {user_time:.2f} seconds")
    except Exception as e:
        print(f"Error during User retrieval: {e}")
        user_time = None

    # Measure Item RAG Tool retrieval time
    start_time = time.time()
    try:
        print("\nQuerying Filtered Item RAG Tool...")
        res = filtered_item_rag_tool._run(search_query="Find reviews about location and food quality.")
        item_time = time.time() - start_time
        print(f"Item Tool Retrieval Time: {item_time:.2f} seconds")
    except Exception as e:
        print(f"Error during Item retrieval: {e}")
        item_time = None

    # Measure Review RAG Tool retrieval time
    start_time = time.time()
    try:
        print("\nQuerying Filtered Review RAG Tool...")
        res = filtered_review_rag_tool._run(search_query="Detailed customer comments on food quality.")
        review_time = time.time() - start_time
        print(f"Review Tool Retrieval Time: {review_time:.2f} seconds")
    except Exception as e:
        print(f"Error during Review retrieval: {e}")
        review_time = None

    print("\n=== Benchmarking Complete ===")
    return user_time, item_time, review_time

if __name__ == "__main__":
    run_benchmark()
