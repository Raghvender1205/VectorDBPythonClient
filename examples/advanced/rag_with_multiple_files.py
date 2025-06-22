import os
from typing import Tuple, List, Dict
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
from loguru import logger

from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from vectordb_client import VectorDBClient, VectorDBClientConnectionError, VectorDBClientRequestError, VectorDBVectorStore

load_dotenv(find_dotenv("../.env"))

PDF_DIR = Path("../docs/")
COLLECTION_NAME = "sample"
METRIC = "cosine"
SOURCE_TAG = "PDF document"
EMBED_DIM = 1536

def load_pdfs(dir_path: Path) -> Tuple[List[str], List[Dict]]:
    """
    Walk and load every *.pdf and return 
        - texts         - List[str]
        - metadatas     - List[dict]
    """
    if not dir_path.exists() or not dir_path.is_dir():
        raise FileNotFoundError(f"Directory not found: {dir_path.resolve()}")

    all_texts: List[str] = []
    all_meta: List[dict] = []

    for pdf_path in sorted(dir_path.glob("*.pdf")):
        loader = PyPDFLoader(str(pdf_path))
        docs = loader.lazy_load() # list[Document]

        page_cnt = 0
        for i, doc in enumerate(docs):
            text = (doc.page_content or "").strip()
            if not text:
                continue

            all_texts.append(text)
            all_meta.append(
                {
                    "file_name": pdf_path.name,
                    "page_number": i + 1,
                    "source": SOURCE_TAG
                }
            )
            page_cnt += 1

        logger.info("✓ %s (%d pages)".format(pdf_path.name, page_cnt))

    return all_texts, all_meta

def get_embedding_model():
    """Instantiate the embedding model once."""
    return OpenAIEmbeddings(
        base_url=os.getenv("EMBEDDING_URL"),
        api_key=os.getenv("EMBEDDING_API_KEY"),
        model=os.getenv("EMBEDDING_MODEL_NAME"),
    )

# ------ RAG ----
def main():
    client = VectorDBClient(server_url="http://localhost:8444")

    try:
        texts, metadatas = load_pdfs(PDF_DIR)
    except FileNotFoundError as e:
        logger.error(str(e))
        return

    if not texts:
        logger.warning(f"No PDF pages found in {PDF_DIR.resolve()}")
        return
    logger.info(f"Total pages loaded: {len(texts)}")

    # Create or reuse collection
    try:
        client.create_collection(name=COLLECTION_NAME, dimension=EMBED_DIM, metric=METRIC)
        logger.info(f"Collection created ({COLLECTION_NAME}, dim={EMBED_DIM}, {METRIC})")
    except VectorDBClientRequestError as exc:
        if exc.status_code == 409:
            logger.info("Collection already exists – using it")
        else:
            raise
    except VectorDBClientConnectionError as exc:
        logger.error(f"Cannot reach Vector DB server: {exc}")
        return

    # Embed
    embedding_model = get_embedding_model()
    vectorstore = VectorDBVectorStore.from_texts(
        texts=texts,
        embedding=embedding_model,
        metadatas=metadatas,
        client=client,
        collection_name=COLLECTION_NAME,
        metric=METRIC,
    )
    logger.info(f"Embedded {len(texts)} pages into collection {COLLECTION_NAME}")

    # QA
    llm = ChatOpenAI(
        base_url=os.getenv("LLM_BASE_URL"),
        api_key=os.getenv("LLM_API_KEY"),
        model=os.getenv("LLM_MODEL_NAME"),
    )
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        return_source_documents=True,
    )

    print("\nReady! Ask anything about the PDFs (type 'exit' to quit).")
    try:
        while True:
            q = input("\n► ")
            if q.lower().strip() == "exit":
                break
            resp = qa_chain.invoke({"query": q})
            print("\nAnswer:\n", resp["result"])
            print("\nSources:")
            for doc in resp["source_documents"]:
                pg = doc.metadata.get("page_number")
                fn = doc.metadata.get("file_name")
                print(f" • {fn}, page {pg}")
    except KeyboardInterrupt:
        print("\nBye!")

if __name__ == "__main__":
    main()