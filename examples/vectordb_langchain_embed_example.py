import os
import numpy as np
import logging

from langchain_community.embeddings.huggingface import HuggingFaceBgeEmbeddings
from langchain_community.document_loaders.pdf import PyPDFLoader

from vectordb_client import VectorDBClient, VectorDBClientConnectionError, VectorDBClientRequestError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def chunk_document(pdf_path: str):
    """Load a PDF file and extract text from each page."""
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    return docs

def get_embeddings():
    """Get embedding model"""
    model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    embeddings = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    
    return embeddings


def main():
    """Embed document and try vector search"""
    pdf_path = "document.pdf"
    server_url = "http://127.0.0.1:8444"
    metadata_category = "pdf_document"

    client = VectorDBClient(server_url=server_url)
    if not os.path.exists(pdf_path):
        logger.error(f'PDF file not found at {pdf_path}')
        return 

    # Chunk pdf
    logger.info(f'Loading PDF from {pdf_path}')
    try:
        docs = chunk_document(pdf_path)
        logger.info(f"Loaded {len(docs)} pages from PDF")
    except Exception as e:
        logger.error(f"Error loading PDF: {e}")
        return
    
    # Get embedding model
    embedding_model = get_embeddings()

    # Embed documents
    documents = []
    for idx , doc in enumerate(docs, start=1):
        text = doc.page_content.strip()
        if not text:
            logger.debug(f"Skipping empty page: {e}")
            continue

        # Generate embeddings
        embedding = embedding_model.embed_documents([text])[0]
        if embedding is None:
            logger.error(f"Failed to generate embeddings for page {idx}")
            continue

        # Create unique Id (PDF_ID_PAGE_NUMBER)
        doc_id = idx
        # Metadata can include more information as needed
        metadata = f"{metadata_category} - Page {idx}"

        documents.append({
            "id": doc_id,
            "embedding": embedding,
            "metadata": metadata
        })

    # Add documents to VectorDB in batch
    try:
        success = client.add_documents(documents)
        if success:
            logger.info(f"Added {len(documents)} documents to VectorDB.")
        else:
            logger.error(f"Failed to add documents to VectorDB.")
    except (VectorDBClientConnectionError, VectorDBClientRequestError) as e:
        logger.error(f"Exception when adding documents: {e}") 

    logger.info("All documents processed.")

    # Search 
    while True:
        try:
            question = input("Ask a question (or type 'exit' to quit): ")
            if question.lower() == 'exit':
                break

            # Generate embedding for the query
            query_embedding = embedding_model.embed_documents([question])[0]
            if query_embedding is None:
                logger.error("Failed to generate embeddings for the query.")
                continue

            # Find nearest neighbors
            retrieved_docs = client.search(
                query=query_embedding, 
                n=5, 
                metric="Cosine", 
                metadata_filter=metadata_category
            )

            if retrieved_docs:
                print("\nRelevant Results:")
                for doc in retrieved_docs:
                    print(f"ID: {doc['id']}, Distance: {doc['distance']:.4f}, Metadata: {doc['metadata']}")
                print("\n")
            else:
                print("No documents retrieved.\n")
        
        except KeyboardInterrupt:
            print("\nExiting.")
            break
        except Exception as e:
            logger.error(f"Error during search: {e}")

if __name__ == "__main__":
    main()