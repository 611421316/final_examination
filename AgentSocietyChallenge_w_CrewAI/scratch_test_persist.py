import os
from crewai_tools import JSONSearchTool

rag_config = {
    "vectordb": {
        "provider": "chromadb",
        "config": {
            "dir": "./my_persisted_db"
        }
    }
}

print("Initializing JSONSearchTool with persistent vectordb...")
# We use a dummy json
with open('dummy.json', 'w') as f:
    f.write('[{"name": "test", "val": 123}]')

tool = JSONSearchTool(
    json_path='dummy.json',
    config=rag_config
)

print("Running tool...")
res = tool._run(search_query="test")
print("Result:", res)

print("Check if dir exists:", os.path.exists("./my_persisted_db"))
