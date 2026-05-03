import os
from embedchain import App

rag_config = {
    "embedder": {
        "provider": "huggingface",
        "config": {
            "model": "BAAI/bge-small-en-v1.5"
        }
    },
    "vectordb": {
        "provider": "chromadb",
        "config": {
            "dir": "./data/my_chroma"
        }
    }
}

app = App.from_config(config=rag_config)
print("Embedder provider:", app.embedding_model.__class__.__name__)
