import chromadb

client = chromadb.PersistentClient(path="data/my_chroma")

for c in client.list_collections():
    col = client.get_collection(c.name)
    print(c.name, col.count())