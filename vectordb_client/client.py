import requests
import logging
import time
from typing import Optional, List, Dict

from vectordb_client.exceptions import VectorDBClientConnectionError, VectorDBClientRequestError


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorDBClient:
    def __init__(
        self,
        server_url: str = "http://127.0.0.1:8444",
        timeout: int = 10,
        max_retries: int = 3
    ):
        """
        Initializes the VectorDBClient

        :param server_url: base url of the VectorDB server
        :param timeout: Timeout for HTTP requests in seconds
        :param max_retries: Max number of retry attempts for failed requests
        """
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = requests.Session()
        self.session.headers.update({'Content-Type': 'application/json'})

    def add_document(self, id: int, embedding: List[float], metadata: str, content) -> bool:
        """
        Adds a document to the VectorDB.

        :param id: Unique identifier of the document
        :param embedding: Embedding vector of the document
        :param metadata: Metadata associated with the document
        :return: True if document was added, False otherwise
        """
        url = f"{self.server_url}/add_document"
        payload = {
            "id": id,
            "embedding": embedding,
            "metadata": metadata,
            "content": content
        }

        for attempt in range(1, self.max_retries + 1):
            try:
                logger.debug(f"Attempt {attempt}: Adding document with id {id}")
                response = self.session.post(url, json=payload, timeout=self.timeout)
                if response.status_code == 200:
                    return True
                else:
                    raise VectorDBClientRequestError(response.status_code, response.text)
            except requests.exceptions.RequestException as e:
                if attempt == self.max_retries:
                    raise VectorDBClientConnectionError(f"Failed to add document after {self.max_retries}") from e

        return False

    # TODO: Add async function of this.
    def add_documents(self, documents: List[Dict]) -> bool:
        """
        Adds multiple documents to the VectorDB
        Each document is a directory with keys: id, embedding, metadata.
        """
        url = f"{self.server_url}/add_documents"
        payload = {"documents": documents}

        for attempt in range(1, self.max_retries + 1):
            try:
                logger.debug(f"Attempt {attempt}: Adding {len(documents)} documents")
                response = self.session.post(url, json=payload, timeout=self.timeout)
                if response.status_code == 200:
                    return True
                else:
                    raise VectorDBClientRequestError(response.status_code, response.text)
            except requests.exceptions.RequestException as e:
                if attempt == self.max_retries:
                    logger.error(f"Failed to add documents after {self.max_retries} attempts.")
                    raise VectorDBClientConnectionError(
                        f"Failed to add documents after {self.max_retries} attempts."
                    ) from e
                else:
                    sleep_time = self.backoff_factor * (2 ** (attempt - 1))
                    logger.warning(f"Attempt {attempt} failed for adding documents. Retrying in {sleep_time} seconds...")
                    time.sleep(sleep_time)

        return False


    def search(
        self,
        query: List[float],
        n: int,
        metric: str = "Dot",
        metadata_filter: Optional[str] = None,
    ) -> List[Dict]:
        """
        Finds the neighboring vectors to a query vector

        :param query: Query embedding vector
        :param n: Number of neighbors to retrieve
        :param metric: Distance metric to use ("Euclidean", "Cosine", "Dot")
        :param metadata_filter: Filter for metadata
        :return: A list of neighboring vectors (documents)
        """
        url = f"{self.server_url}/search"
        payload = {
            "query": query,
            "n": n,
            "metric": metric,
            "metadata_filter": metadata_filter
        }

        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.session.post(url, json=payload, timeout=self.timeout)
                if response.status_code == 200:
                    return response.json()
                else:
                    raise VectorDBClientRequestError(response.status_code, response.text)
            except requests.exceptions.RequestException as e:
                if attempt == self.max_retries:
                    raise VectorDBClientConnectionError(f"Failed to search after {self.max_retries}") from e

        return []
