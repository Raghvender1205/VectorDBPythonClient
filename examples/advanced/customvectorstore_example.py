import os
import logging
from pathlib import Path
from typing import List
from dotenv import load_dotenv, find_dotenv

from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from vectordb_client import (
    VectorDBClient,
    VectorDBClientConnectionError,
    VectorDBClientRequestError,
    VectorDBVectorStore,
)

load_dotenv(find_dotenv("../.env"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PDF_PATH = Path("../document.pdf")
SERVER_URL = "http://127.0.0.1:8444"
COLLECTION_NAME = "sample_collection"
METRIC = "cosine"           # or "dot", "euclidean"
SOURCE_TAG = "PDF document" # extra metadata tag
EMBED_DIM = 1536


def chunk_document(path: Path) -> List[str]:
    """Load a PDF file and extract text from each page."""
    loader = PyPDFLoader(str(path))
    docs = loader.load()
    
    return [d.page_content.strip() for d in docs if d.page_content.strip()]


def get_embedding_model():
    """Get embedding model"""
    embedding_model = OpenAIEmbeddings(
        base_url=os.getenv("EMBEDDING_URL"),
        api_key=os.getenv("EMBEDDING_API_KEY"),
        model=os.getenv("EMBEDDING_MODEL_NAME"),
    )

    return embedding_model


def main():
    """Embed document and perform vector search using VectorDBVectorStore and LangChain"""
    client = VectorDBClient(server_url=SERVER_URL)

    if not PDF_PATH.exists():
        logger.error("PDF file not found: %s", PDF_PATH)
        return
    
    # Embedding model
    embedding = get_embedding_model()

    llm = ChatOpenAI(
        base_url=os.getenv("LLM_BASE_URL"),
        api_key=os.getenv("LLM_API_KEY"),
        model=os.getenv("LLM_MODEL_NAME"),
    )
    
    # Read PDF
    texts = chunk_document(PDF_PATH)
    if not texts:
        logger.error("PDF file not found: %s", PDF_PATH)
        return
    logger.info("Loaded %d pages from PDF", len(texts))

    # Create Collection
    try:
        client.create_collection(
            name=COLLECTION_NAME,
            dimension=EMBED_DIM,
            metric=METRIC,
        )
        logger.info("Collection created (%s, dim=%d, %s)", COLLECTION_NAME, EMBED_DIM, METRIC)
    except VectorDBClientRequestError as exc:
        if exc.status_code == 409:
            logger.info("Collection already exists, using it")
        else:
            raise
    except VectorDBClientConnectionError as exc:
        logger.error("Cannot reach Vector DB server: %s", exc)
        return
    
    # Metadata Per page
    metadatas = [
        {"page_number": i + 1, "source": SOURCE_TAG}
        for i in range(len(texts))
    ]

    # Insert to vectordb
    vectorstore = VectorDBVectorStore.from_texts(
        texts=texts,
        embedding=embedding,
        metadatas=metadatas,
        client=client,
        collection_name=COLLECTION_NAME,
        metric=METRIC,
    )
    logger.info("PDF embedded and stored (collection=%s)", COLLECTION_NAME)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        return_source_documents=True,
    )

    print("\nLoaded! Ask questions (type 'exit' to quit).")
    try:
        while True:
            q = input("\n► ")
            if q.lower().strip() == "exit":
                break
            resp = qa_chain.invoke(q)
            print("\nAnswer:\n", resp["result"])
            print("\nSources:")
            for doc in resp["source_documents"]:
                pg = doc.metadata.get("page_number")
                print(f" • page {pg}")
    except KeyboardInterrupt:
        print("\nBye!")

if __name__ == "__main__":
    main()
