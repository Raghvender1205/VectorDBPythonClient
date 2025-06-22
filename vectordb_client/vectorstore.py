import logging
import json
from typing import List, Optional, Dict, Any, Sequence
from typing_extensions import Type

from langchain.vectorstores.base import VectorStore
from langchain.schema import Document

from vectordb_client import (
    VectorDBClient,
    VectorDBClientConnectionError,
    VectorDBClientRequestError,
)

logger = logging.getLogger(__name__)


class VectorDBVectorStore(VectorStore):
    """Custom VectorStore implementation for VectorDBClient"""

    def __init__(
        self,
        client: VectorDBClient,
        collection_name: str,
        embedding_model: Any,
        metric: str = "cosine",
    ):
        """
        Initializes the VectorDBVectorStore

        :param client: VectorDBClient
        :param collection_name: The name of the collection to use.
        :param embedding_model: Embedding model instance.
        :param metric: Distance Metric to use
        """
        self.client = client
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.metric = metric  # "cosine" | "euclidean" | "dot"

    @property
    def _vectorstore_type(self) -> str:
        return "vectordbvectorstore"
    
    def _ensure_collection(self, dim: int) -> None:
        """
        Lazy Create collection 
        """
        if self.client.get_collection(self.collection_name):
            return # already exists
        
        logger.info(
            "Creating collection '%s' (dim=%d metric=%s)",
            self.collection_name,
            dim,
            self.metric
        )
        try:
            self.client.create_collection(
                name=self.collection_name,
                dimension=dim,
                metric=self.metric
            )
        except VectorDBClientRequestError as exc:
            if exc.status_code == 400:
                logger.debug("Collection already exists, continuing")
            else:
                raise

    def add_texts(
        self, texts: List[str], metadatas: Optional[Sequence[dict]] = None, **kwargs: Any
    ) -> List[str]:
        """
        Add multiple texts to the vector store

        :param texts: List of texts to add
        :param metadatas: Optional list of metadata dictionaries
        :return: List of document IDs
        """
        if not texts:
            return []
        
        documents: List[Dict] = []
        embed_dim: Optional[int] = None

        for idx, txt in enumerate(texts):
            try:
                vec = self.embedding_model.embed_documents([txt])[0]
            except Exception as exc: # pylint: disable=broad-except
                logger.error("Embedding failed for item %d: %s", idx, exc)
                continue

            if vec is None:
                logger.error("Embedding model returned None for item %d", idx)
                continue

            if embed_dim is None:
                embed_dim = len(vec)

            meta = (
                metadatas[idx].copy() if metadatas and idx < len(metadatas) else {}
            )
            meta.update(kwargs.get("additional_metadata", {}))

            documents.append(
                {
                    "embedding": vec,
                    "metadata": json.dumps(meta),  # server expects string
                    "content": txt,
                }
            )
        
        if not documents:
            logger.warning("No documents were successfully embedded")
            return []
        
        # Ensure collection 
        self._ensure_collection(embed_dim)

        try:
            ids = self.client.add_documents(documents, self.collection_name)
            if not ids:
                logger.warning("Server inserted 0 of %d documents", len(documents))
                return []

            logger.info("Inserted %d / %d docs", len(ids), len(documents))
            
            return [str(i) for i in ids]

        except (VectorDBClientConnectionError, VectorDBClientRequestError) as exc:
            logger.error("add_documents failed: %s", exc)
            
            return []

    def _embed_query(self, query: str) -> Optional[List[float]]:
        """
        Embed the input query
        """
        try:
            vec = self.embedding_model.embed_documents([query])[0]
            
            return vec
        except Exception as exc:  # pylint: disable=broad-except
            logger.error("Query embedding failed: %s", exc)
            
            return None

    def similarity_search(
        self, query: str, k: int = 4, **_: Any
    ) -> List[Document]:
        """
        Performs a similarity search

        :param query: The query text
        :param k: Number of top results to return
        :return: List of documents
        """
        vec = self._embed_query(query)
        if vec is None:
            return []

        try:
            hits = self.client.search(
                query=vec,
                n=k,
                collection_name=self.collection_name,
            )
        except (VectorDBClientConnectionError, VectorDBClientRequestError) as exc:
            logger.error("Search failed: %s", exc)
            return []
        
        docs: List[Document] = []
        for hit in hits:
            docs.append(
                Document(
                    page_content=hit.get("content", ""),
                    metadata=json.loads(hit.get("metadata", "{}")),
                    score=hit.get("distance", 0.0),
                )
            )
        
        return docs

    def similarity_search_with_score(
        self, query: str, k: int = 4, **kwargs: Any
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
        *,
        client: VectorDBClient = None,
        collection_name: str = "",
        metric: str = "cosine",
        **kwargs: Any,
    ) -> "VectorDBVectorStore":
        """
        Create a VectorDBVectorStore from a list of texts.

        Example
            store = VectorDBVectorStore.from_texts(
                texts,
                embedding=openai_embedder,
                client=my_client,
                collection_name="my_coll",
                metric="dot",
            )
        """
        store = cls(
            client=client,
            collection_name=collection_name,
            embedding_model=embedding,
            metric=metric,
        )
        ids = store.add_texts(texts, metadatas, **kwargs)
        if len(ids) < len(texts):
            logger.warning("%d texts failed to insert", len(texts) - len(ids))
        
        return store