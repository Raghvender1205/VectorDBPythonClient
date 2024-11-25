import logging
import json
from typing import List, Optional, Dict, Any
from typing_extensions import Type

from langchain.vectorstores.base import VectorStore
from langchain.schema import Document

from vectordb_client import (
    VectorDBClient,
    VectorDBClientConnectionError,
    VectorDBClientRequestError
)

logger = logging.getLogger(__name__)


class VectorDBVectorStore(VectorStore):
    """Custom VectorStore implementation for VectorDBClient"""
    
    def __init__(
        self,
        client: VectorDBClient,
        collection_name: str,
        embedding_model: Any,  
    ):
        """
        Initializes the VectorDBVectorStore

        :param client: VectorDBClient
        :param collection_name: The name of the collection to use.
        :param embedding_model: Embedding model instance.
        """
        self.client = client
        self.collection_name = collection_name
        self.embedding_model = embedding_model

    
    @property
    def _vectorstore_type(self) -> str:
        return "vectordbvectorstore"
    
    def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any
    ) -> List[str]:
        """
        Add multiple texts to the vector store

        :param texts: List of texts to add
        :param metadatas: Optional list of metadata dictionaries
        :return: List of document IDs
        """
        documents = []
        for idx, text in enumerate(texts):
            metadata = metadatas[idx] if metadatas and idx < len(metadatas) else {}
            try:
                # Generate embedding using the embedding model's method
                embedding = self.embedding_model.embed_documents([text])[0]
                if embedding is None:
                    logger.error(f"Failed to generate embedding for text index {idx}")
                    continue
            except Exception as e:
                logger.error(f"Exception during embedding generation for text index {idx}: {e}")
                continue

            doc_id = idx + 1  # Use integer IDs starting from 1
            metadata.update(kwargs.get("additional_metadata", {}))
            # Serialize metadata to JSON string as Rust backend expects a string
            metadata_str = json.dumps(metadata)
            documents.append({
                "id": doc_id,
                "embedding": embedding,
                "metadata": metadata_str,
                "content": text
            })

        if not documents:
            logger.warning("No documents were successfully embedded and added.")
            return []
        
        try:
            success = self.client.add_documents(documents, self.collection_name)
            if success:
                logger.info(f"Added {len(documents)} documents to collection '{self.collection_name}'.")
                return [str(doc["id"]) for doc in documents]  # Return IDs as strings for consistency
            else:
                logger.error(f"Failed to add documents to collection '{self.collection_name}'.")
                return []
        except (VectorDBClientConnectionError, VectorDBClientRequestError) as e:
            logger.error(f"Exception when adding documents: {e}")
            return []
        

    def similarity_search(
        self, 
        query: str,
        k: int = 4,
        **kwargs: Any
    ) -> List[Document]:
        """
        Performs a similarity search

        :param query: The query text
        :param k: Number of top results to return
        :return: List of documents
        """
        try:
            query_embedding = self.embedding_model.embed_documents([query])[0]
            if query_embedding is None:
                logger.error("Failed to generate embedding for the query")
                return []
        except Exception as e:
            logger.error(f"Exception during embedding generation for query: {e}")
            return []
        
        try:
            retrieved_docs = self.client.search(
                query=query_embedding,
                n=k,
                metric="Cosine", # TODO: Add kwargs for other distance metric options
                collection_name=self.collection_name
            )
            documents = []
            for doc in retrieved_docs:
                # Assuming 'distance' is returned; adjust if different
                documents.append(Document(
                    page_content=doc.get("content", ""),
                    metadata=json.loads(doc.get("metadata", "{}")),  # Deserialize metadata back to dict
                    score=doc.get("distance", 0.0)
                ))
            
            return documents
        except (VectorDBClientConnectionError, VectorDBClientRequestError) as e:
            logger.error(f"Exception during similarity search: {e}")
            return []
        
    def similarity_search_with_score(
        self, 
        query: str,
        k: int = 4,
        **kwargs: Any
    ) -> List[Document]:
        """
        Performs a similarity search and returns documents with scores.

        :param query: The query text.
        :param k: Number of top results to return.
        :return: List of Documents with scores.
        """
        # This method is similar to similarity_search and can be customized if needed
        return self.similarity_search(query, k, **kwargs)
        

    @classmethod
    def from_texts(
        cls: Type["VectorDBVectorStore"],
        texts: List[str],
        embedding: Any,
        metadatas: Optional[List[dict]] = None,
        client: VectorDBClient = None,
        collection_name: str = "",
        **kwargs: Any,
    ) -> 'VectorDBVectorStore':
        """
        Create a VectorDBVectorStore from a list of texts.

        :param texts: List of texts to add.
        :param embedding: Embedding model instance.
        :param metadatas: Optional list of metadata dictionaries.
        :param client: Instance of VectorDBClient.
        :param collection_name: Name of the collection to use.
        :param kwargs: Additional arguments.
        :return: An instance of VectorDBVectorStore.
        """
        if client is None:
            raise ValueError("VectorDBClient instance must be provided via kwargs['client']")
        if not collection_name:
            raise ValueError("Collection name must be provided via kwargs['collection_name']")
        if embedding is None:
            raise ValueError("Embedding model must be provided.")

        store = cls(client=client, collection_name=collection_name, embedding_model=embedding)
        store.add_texts(
            texts=texts,
            metadatas=metadatas,
            **kwargs
        )
        return store
