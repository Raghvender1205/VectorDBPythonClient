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

    def _retry_loop(self, verb: str, url: str, **req_kwargs):
        """
        Generic retry wrapper
        """
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self.session.request(
                    method=verb, url=url, timeout=self.timeout, **req_kwargs
                )

                return resp
            except requests.exceptions.RequestException as exc:
                logger.error("RequestException on attempt %d: %s", attempt, exc)
                if attempt == self.max_retries:
                    raise VectorDBClientConnectionError(
                        f"Request to {url} failed after {self.max_retries} tries"
                    ) from exc
                sleep_s = self.backoff_factor * (2 ** (attempt - 1))
                logger.debug("Retrying in %.1fs ...", sleep_s)
                time.sleep(sleep_s)

    def create_collection(
        self, 
        name: str,
        dimension: int,
        metric: str = "cosine"
    ) -> Optional[Collection]:
        """
        Creates a new collection in the vectordb

        :param name: Name of the collection
        :return: Collection object if created, None otherwise 
        """
        url = f"{self.server_url}/create_collection"
        payload = {"name": name, "dimension": dimension, "metric": metric}

        resp = self._retry_loop("POST", url, json=payload)
        if resp.status_code == 200:
            return Collection.from_dict(resp.json())
        if resp.status_code == 409:
            raise VectorDBClientRequestError(409, f"Collection '{name}' exists")
        
        raise VectorDBClientRequestError(resp.status_code, resp.text)
    
    def get_collection(self, name: str) -> Optional[Collection]:
        """
        Get collection from VectorDB
        """
        url = f"{self.server_url}/collections/{name}"
        resp = self._retry_loop("GET", url)
        if resp.status_code == 200:
            return Collection.from_dict(resp.json())
        if resp.status_code == 404:
            return None
        
        raise VectorDBClientRequestError(resp.status_code, resp.text)
    
    def list_collections(self) -> List[Collection]:
        """
        List all the collections from VectorDB
        """
        url = f"{self.server_url}/collections"
        resp = self._retry_loop("GET", url)
        
        if resp.status_code == 200:
            return [Collection.from_dict(obj) for obj in resp.json()]
        
        raise VectorDBClientRequestError(resp.status_code, resp.text)
        

    def add_document(
        self, 
        embedding: List[float], 
        metadata: str, 
        content: str,
        collection_name: str,
        doc_id: Optional[int] = None
    ) -> int:
        """
        Adds a document to the VectorDB.

        Returns server-generated ID if you have not supplied one
        """
        url = f"{self.server_url}/add_document"
        payload = {
            "embedding": embedding,
            "metadata": metadata,
            "content": content,
            "collection_name": collection_name,
        }
        if doc_id is not None:
            payload["id"] = doc_id

        resp = self._retry_loop("POST", url, json=payload)
        if resp.status_code == 200:
            return resp.json()["id"]
        
        raise VectorDBClientRequestError(resp.status_code, resp.text)
    

    def add_documents(self, documents: List[Dict], collection_name: str) -> List[int]:
        """
        Adds multiple documents to the VectorDB

        Returns list of IDs that were *actually* inserted.
        """
        # ensure each doc contains its collection
        for d in documents:
            d["collection_name"] = collection_name
            if d.get("id") is None:  # trim empty id
                d.pop("id", None)

        url = f"{self.server_url}/add_documents"
        resp = self._retry_loop("POST", url, json={"documents": documents})

        if resp.status_code == 200:
            return [d["id"] for d in resp.json()["documents"]]

        if resp.status_code == 207:
            body = resp.json()
            ids = [d["id"] for d in body.get("documents", []) if d["status"] == "success"]
            for err in body.get("errors", []):
                logger.warning("Server insert error: %s", err)
            return ids

        raise VectorDBClientRequestError(resp.status_code, resp.text)

    # TODO: Add metadata filtering
    def search(
        self,
        query: List[float],
        n: int,
        collection_name: str,
    ) -> List[Dict]:
        """
        Similarity Search
        """
        url = f"{self.server_url}/search"
        payload = {
            "query": query,
            "n": n,
            "collection_name": collection_name,
        }

        resp = self._retry_loop("POST", url, json=payload)
        if resp.status_code == 200:
            return resp.json()
        
        raise VectorDBClientRequestError(resp.status_code, resp.text)