from vectordb_client import VectorDBClient

client = VectorDBClient(server_url="http://127.0.0.1:8444")

collection_name = "pdf_documents"
collection = client.create_collection(collection_name)

if collection:
    print(f"Collection created: ID={collection.id}, Name='{collection.name}'")
else:
    print(f"Collection '{collection_name}' already exists.")