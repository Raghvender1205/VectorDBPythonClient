import logging
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
    ):
        """
        Initializes the VectorDBVectorStore

        :param client: VectorDBClient
        :param collection_name: The name of the collection to use.
        """
        self.client = client
        self.collection_name = collection_name

    
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
            embedding = kwargs.get("embedding_model")(text)
            if embedding is None:
                logger.error(f"Failed to generate embedding for text index {idx}")
                continue

            doc_id = kwargs.get("doc_id_prefix", "") + str(idx + 1)
            metadata.update(kwargs.get("additional_metadata", {}))
            documents.append({
                "id": doc_id,
                "embedding": embedding,
                "metadata": metadata,
                "content": text
            })

        try:
            success = self.client.add_documents(documents, self.collection_name)
            if success:
                logger.info(f"Added {len(documents)} documents to collection '{self.collection_name}'.")
                return [doc["id"] for doc in documents]
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
        embedding_model = kwargs.get("embedding_model")
        if embedding_model is None:
            raise ValueError("Embedding model must be provided")
        
        query_embedding = embedding_model.embed_documents([query])[0]
        if query_embedding is None:
            logger.error("Failed to generate embedding for the query")
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
                documents.append(Document(
                    page_content=doc.get("content", ""),
                    metadata=doc.get("metadata", {}),
                    score=doc.get("distance", 0.0)
                ))
            
            return documents
        except (VectorDBClientConnectionError, VectorDBClientRequestError) as e:
            logger.error(f"Exception during similarity search with score: {e}")
            
            return []
        
    # TODO: Score threshold
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
        embedding_model = kwargs.get("embedding_model")
        if embedding_model is None:
            raise ValueError("Embedding model must be provided")
        
        query_embedding = embedding_model.embed_documents([query])[0]
        if query_embedding is None:
            logger.error("Failed to generate embedding for the query")
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
                documents.append(Document(
                    page_content=doc.get("content", ""),
                    metadata=doc.get("metadata", {}),
                    score=doc.get("distance", 0.0)
                ))
            
            return documents
        except (VectorDBClientConnectionError, VectorDBClientRequestError) as e:
            logger.error(f"Exception during similarity search with score: {e}")
            
            return []
        

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

        store = cls(client=client, collection_name=collection_name)
        store.add_texts(
            texts=texts,
            metadatas=metadatas,
            embedding_model=embedding,
            **kwargs
        )
        return store