import requests
import logging
import time
from typing import Optional, List, Dict

from vectordb_client.exceptions import (
    VectorDBClientConnectionError, 
    VectorDBClientRequestError,
    VectorDBClientValidationError
)
from vectordb_client.models import Collection


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorDBClient:
    def __init__(
        self,
        server_url: str = "http://127.0.0.1:8444",
        timeout: int = 10,
        max_retries: int = 3,
        backoff_factor: float = 0.5
    ):
        """
        Initializes the VectorDBClient

        :param server_url: base url of the VectorDB server
        :param timeout: Timeout for HTTP requests in seconds
        :param max_retries: Max number of retry attempts for failed requests
        :param backoff_factor: Exponential backoff between retries
        """
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.session = requests.Session()
        self.session.headers.update({'Content-Type': 'application/json'})


    def create_collection(self, collection_name: str) -> Optional[Collection]:
        """
        Creates a new collection in the vectordb

        :param collection_name: Name of the collection
        :return: Collection object if created, None otherwise 
        """
        url = f"{self.server_url}/create_collection"
        payload = {"name": collection_name}

        for attempt in range(1, self.max_retries + 1):
            try: 
                logger.debug(f"Attempt {attempt}: Creating collection '{collection_name}'")
                response = self.session.post(url, json=payload, timeout=self.timeout)
                if response.status_code == 200:
                    collection_data = response.json()
                    return Collection.from_dict(collection_data)
                elif response.status_code == 409:
                    logger.warning(f"Collection '{collection_name}' already exists.")
                    return None
                else:
                    raise VectorDBClientRequestError(response.status_code, response.text)
            except requests.exceptions.RequestException as e:
                logger.error(f"RequestException on attempt {attempt}: {e}")
                if attempt == self.max_retries:
                    raise VectorDBClientConnectionError(
                        f"Failed to create collection '{collection_name}' after {self.max_retries} attempts."
                    ) from e
                sleep_time = self.backoff_factor * (2 ** (attempt - 1))
                logger.info(f"Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)

    def add_document(self, id: int, embedding: List[float], metadata: str, content: str, collection_name: str) -> bool:
        """
        Adds a document to the VectorDB.

        :param id: Unique identifier of the document
        :param embedding: Embedding vector of the document
        :param metadata: Metadata associated with the document
        param content: Content of the document
        :param collection_name: Name of the collection to add the document
        :return: True if document was added, False otherwise
        """
        url = f"{self.server_url}/add_document"
        payload = {
            "id": id,
            "embedding": embedding,
            "metadata": metadata,
            "content": content,
            "collection_name": collection_name
        }

        for attempt in range(1, self.max_retries + 1):
            try:
                logger.debug(f"Attempt {attempt}: Adding document with id {id} to collection '{collection_name}'")
                response = self.session.post(url, json=payload, timeout=self.timeout)
                if response.status_code == 200:
                    return True
                else:
                    raise VectorDBClientRequestError(response.status_code, response.text)
            except requests.exceptions.RequestException as e:
                logger.error(f"RequestException on attempt {attempt}: {e}")
                if attempt == self.max_retries:
                    raise VectorDBClientConnectionError(
                        f"Failed to add document {id} after {self.max_retries} attempts."
                    ) from e
                sleep_time = self.backoff_factor * (2 ** (attempt - 1))
                logger.info(f"Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)

        return False
    

    def add_documents(self, documents: List[Dict], collection_name: str) -> bool:
        """
        Adds multiple documents to the VectorDB
        Each document is a directory with keys: id, embedding, metadata.

        :param documents: List of document dictionaries with keys: id, embedding, metadata, content
        :param collection_name: Name of the collection to add the documents to
        :return: True if all documents were added successfully, False otherwise
        """
        url = f"{self.server_url}/add_documents"
        # Ensure all documents have the collection_name
        for doc in documents:
            doc['collection_name'] = collection_name

        payload = {"documents": documents}

        for attempt in range(1, self.max_retries + 1):
            try:
                logger.debug(f"Attempt {attempt}: Adding {len(documents)} documents to collection '{collection_name}'")
                response = self.session.post(url, json=payload, timeout=self.timeout)
                if response.status_code == 200:
                    return True
                else:
                    raise VectorDBClientRequestError(response.status_code, response.text)
            except requests.exceptions.RequestException as e:
                logger.error(f"RequestException on attempt {attempt}: {e}")
                if attempt == self.max_retries:
                    raise VectorDBClientConnectionError(
                        f"Failed to add documents after {self.max_retries} attempts."
                    ) from e
                sleep_time = self.backoff_factor * (2 ** (attempt - 1))
                logger.info(f"Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)

        return False

    # TODO: Add metadata filtering
    def search(
        self,
        query: List[float],
        n: int,
        metric: str = "Dot",
        # metadata_filter: Optional[str] = None,
        collection_name: str = "",
    ) -> List[Dict]:
        """
        Finds the neighboring vectors to a query vector

        :param query: Query embedding vector
        :param n: Number of neighbors to retrieve
        :param metric: Distance metric to use ("Euclidean", "Cosine", "Dot")
        :param metadata_filter: Filter for metadata
        :param collection_name: Name of the collection to search within
        :return: A list of neighboring vectors (documents)
        """
        url = f"{self.server_url}/search"
        payload = {
            "query": query,
            "n": n,
            "metric": metric,
            # "metadata_filter": metadata_filter
            "collection_name": collection_name
        }

        for attempt in range(1, self.max_retries + 1):
            try:
                logger.debug(f"Attempt {attempt}: Searching in collection '{collection_name}' with metric '{metric}'")
                response = self.session.post(url, json=payload, timeout=self.timeout)
                if response.status_code == 200:
                    return response.json()
                else:
                    raise VectorDBClientRequestError(response.status_code, response.text)
            except requests.exceptions.RequestException as e:
                logger.error(f"RequestException on attempt {attempt}: {e}")
                if attempt == self.max_retries:
                    raise VectorDBClientConnectionError(
                        f"Failed to search after {self.max_retries} attempts."
                    ) from e
                sleep_time = self.backoff_factor * (2 ** (attempt - 1))
                logger.info(f"Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)

        return []
