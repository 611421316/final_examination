import time
from datetime import datetime
import os
from dotenv import load_dotenv
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

def benchmark_single_tool(name: str, json_path: str, run_id: int):
    print(f"\n=== Benchmarking {name} Database ===")
    
    # 1. Measure initialization (this is where CrewAI spends heavy CPU on chunking & ChromaDB writes)
    print(f"[{name}] Starting Tool Initialization & Fresh Indexing (HEAVY CPU computation)...")
    init_start = time.time()
    
    rag_tool = JSONSearchTool(
        json_path=json_path,
        collection_name=f'benchmark_true_fresh_index_{name}_{run_id}',
        config=rag_config
    )
    
    init_end = time.time()
    print(f"[{name}] Initialization & Indexing Time: {init_end - init_start:.2f} seconds")
    
    # 2. Measure retrieval speed after warm-up (pure ChromaDB query)
    print(f"[{name}] Starting RAG Retrieval test...")
    retrieval_start = time.time()
    
    try:
        res = rag_tool._run(search_query="Find relevant information")
        retrieval_end = time.time()
        print(f"[{name}] Pure Retrieval Time: {retrieval_end - retrieval_start:.2f} seconds")
    except Exception as e:
        print(f"Error during {name} retrieval: {e}")

def run_indexing_benchmark():
    # Use current timestamp as run_id to ensure ChromaDB cannot match any existing collection snapshots
    run_id = int(time.time())
    print("=== Starting True Vector Indexing Latency Benchmark ===")
    print("We are now measuring the exact time spent ON INITIALIZING the JSONSearchTool.\n")
    
    benchmark_single_tool('Filtered_User', 'data/filtered_user.json', 1)
    benchmark_single_tool('Filtered_Item', 'data/filtered_item.json', 1)
    benchmark_single_tool('Filtered_Review', 'data/test_review.json', 1)

if __name__ == "__main__":
    run_indexing_benchmark()
