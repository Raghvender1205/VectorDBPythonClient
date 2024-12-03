import asyncio
from vectordb_client.async_client import AsyncVectorDBClient

async def main():
    # Initialize the async client
    async_client = AsyncVectorDBClient(server_url="http://127.0.0.1:8444")

    # Create a new collection
    collection_name = "pdf_documents_async"
    collection = await async_client.acreate_collection(collection_name)

    if collection:
        print(f"Collection created: ID={collection.id}, Name='{collection.name}'")
    else:
        print(f"Collection '{collection_name}' already exists.")

    # Add a document asynchronously
    success = await async_client.aadd_document(
        id=4,
        embedding=[0.2, 0.3, 0.4],
        metadata="Appendix",
        content="Content of the appendix.",
        collection_name=collection_name
    )

    if success:
        print(f"Document 4 added successfully to collection '{collection_name}'.")
    else:
        print(f"Failed to add document 4 to collection '{collection_name}'.")

    # Perform an asynchronous search
    results = await async_client.search(
        query=[0.2, 0.3, 0.4],
        n=1,
        metric="Euclidean",
        metadata_filter="pdf_documents_async",
        collection_name=collection_name
    )

    print("Asynchronous Search Results:")
    for doc in results:
        print(doc)

    # Close the async client
    await async_client.aclose()

# Run the async main function
asyncio.run(main())
