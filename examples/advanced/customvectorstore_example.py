import os
import logging

from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_ollama import ChatOllama
from vectordb_client import VectorDBClient, VectorDBClientConnectionError, VectorDBClientRequestError, VectorDBVectorStore

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
    embeddings = HuggingFaceBgeEmbeddings(model_name=model_name, model_kwargs={'device': 'cpu'})
    return embeddings


def main():
    """Embed document and perform vector search using VectorDBVectorStore and LangChain"""
    pdf_path = "../document.pdf"
    server_url = "http://127.0.0.1:8444"
    metadata_category = "pdf_document"
    collection_name = "sample"

    client = VectorDBClient(server_url=server_url)
    if not os.path.exists(pdf_path):
        logger.error(f'PDF file not found at {pdf_path}')
        return 

    # Create or get collection
    logger.info(f'Creating or retrieving collection "{collection_name}"')
    try:
        collection = client.create_collection(collection_name)
        if collection:
            logger.info(f"Collection created: ID={collection.id}, Name='{collection.name}'")
        else:
            # If collection already exists, it has been retrieved by the client
            collection = client.get_collection(collection_name)
            if collection:
                logger.info(f"Collection already exists: ID={collection.id}, Name='{collection.name}'")
            else:
                logger.error(f"Collection '{collection_name}' exists but failed to retrieve details.")
                return
    except (VectorDBClientConnectionError, VectorDBClientRequestError) as e:
        logger.error(f"Exception when creating/retrieving collection: {e}")
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

    # Prepare texts and metadatas
    texts = []
    metadatas = []
    for idx, doc in enumerate(docs, start=1):
        text = doc.page_content.strip()
        if not text:
            logger.debug(f"Skipping empty page {idx}")
            continue

        # Metadata can include more information as needed
        metadata = {"category": metadata_category, "page_number": idx}
        texts.append(text)
        metadatas.append(metadata)

    # Initialize the custom VectorStore using 'from_texts'
    if not texts:
        logger.warning("No texts to add after processing PDF.")
    else:
        try:
            vectordb_store = VectorDBVectorStore.from_texts(
                texts=texts,
                embedding=embedding_model,
                metadatas=metadatas,
                client=client,
                collection_name=collection_name,
                additional_metadata={"source": "PDF Document"}
            )
            logger.info(f"Added {len(texts)} documents to collection '{collection_name}'.")
        except Exception as e:
            logger.error(f"Exception when adding texts: {e}") 
            return

    logger.info("All documents processed and added to VectorDB.")

    # Initialize the QA chain with an actual LLM
    try:
        llm = ChatOllama(model="llama3.1:8b", temperature=0, cache=False)  # Ensure ChatOllama is correctly set up
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectordb_store.as_retriever(),  # Ensure 'as_retriever' method is available
        )
    except Exception as e:
        logger.error(f"Exception when initializing QA chain: {e}")
        return

    # Interactive search
    logger.info("You can now ask questions. Type 'exit' to quit.")
    while True:
        try:
            question = input("Ask a question (or type 'exit' to quit): ")
            if question.lower() == 'exit':
                break

            # Update to use the 'invoke' method to avoid deprecation warnings
            response = qa_chain.invoke(question)
            print("\nAnswer:")
            print(response)
            print("\n")
        
        except KeyboardInterrupt:
            print("\nExiting.")
            break
        except Exception as e:
            logger.error(f"Error during search: {e}")


if __name__ == "__main__":
    main()
