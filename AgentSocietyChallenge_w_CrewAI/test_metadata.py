import chromadb
from chromadb.config import Settings
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent
_CHROMA_DIR = str(_PROJECT_ROOT / "data" / "my_chroma")
_USER_COLLECTION = "benchmark_true_fresh_index_Filtered_User_3"

client = chromadb.PersistentClient(
    path=_CHROMA_DIR,
    settings=Settings(anonymized_telemetry=False),
)

col = client.get_collection(_USER_COLLECTION)
results = col.get(
    limit=2,
    include=["documents", "metadatas"],
)
print("First 2 documents metadatas:")
print(results["metadatas"])
print("\nFirst 2 documents texts:")
print(results["documents"])
