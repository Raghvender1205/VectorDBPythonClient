# VectorDB Python Client
<div align="center">
    <img src="assets/logo.png" width="400" />
</div>

Python SDK for [VectorDB](https://github.com/Raghvender1205/VectorDB)

Currently, it can embed documents using `Langchain document loader and huggingface embeddings` then search through. See [here](https://github.com/Raghvender1205/VectorDBPythonClient/blob/master/examples/vectordb_langchain_embed_example.py)

## TODO
1. Use Type Annotations and Data Validations
2. Search threshold for more relevant results
3. Logging
4. API Keys Middleware
5. Handle edge cases
6. Refine the examples
7. Implement pagination for search results
8. Delete document endpoint 
9. Allow to add similar documents in different collections
10. Metadata filtering inside a collection docs
11. Test Async client and examples. See [here](https://github.com/Raghvender1205/VectorDBPythonClient/tree/master/examples/async_client_test.py)

Implement `CustomVectorStore` for compatibility with `langchain`, `langgraph` etc.

https://github.com/langchain-ai/langchain/discussions/17238