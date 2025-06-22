import os
import logging
from pathlib import Path
from loguru import logger
from typing import List, Tuple, Dict

from dotenv import load_dotenv, find_dotenv

from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from vectordb_client import (
    VectorDBClient, VectorDBVectorStore,
    VectorDBClientRequestError, VectorDBClientConnectionError,
)

from ragas import SingleTurnSample, EvaluationDataset, evaluate
from ragas.llms.base import LangchainLLMWrapper
from ragas.embeddings.base import LangchainEmbeddingsWrapper
from ragas.metrics import (
    context_precision,
    context_recall,
    answer_relevancy,
    answer_correctness,
    faithfulness
)

load_dotenv(find_dotenv("../.env"))

PDF_DIR         = Path("../docs")
SERVER_URL      = "http://127.0.0.1:8444"
COLLECTION_NAME = "sample_collection"
EMBED_DIM       = 1536
METRIC          = "cosine"
SOURCE_TAG      = "PDF document"


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

        logger.info(f"✓ {pdf_path.name} {page_cnt} pages")

    return all_texts, all_meta


def get_embedding_model():
    """Instantiate the embedding model once."""
    return OpenAIEmbeddings(
        base_url=os.getenv("EMBEDDING_URL"),
        api_key=os.getenv("EMBEDDING_API_KEY"),
        model=os.getenv("EMBEDDING_MODEL_NAME"),
    )

def chat_llm():
    return ChatOpenAI(
        base_url=os.getenv("LLM_BASE_URL"),
        api_key=os.getenv("LLM_API_KEY"),
        model=os.getenv("LLM_MODEL_NAME"),
    )


def main():
    client = VectorDBClient(server_url=SERVER_URL)

    evaluator_llm = LangchainLLMWrapper(chat_llm())
    evaluator_emb = LangchainEmbeddingsWrapper(
        get_embedding_model()
    )

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

    qa_chain = RetrievalQA.from_chain_type(
        llm=chat_llm(),
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        return_source_documents=True,
    )

    # Ground truth pairs for eval
    gt_pairs = [
        ("What does PGI mean?", "Programmable Gradient Information"),
        ("Who wrote the paper?", "Chung Yuan Christian University")
    ]

    samples = []
    for q, truth in gt_pairs:
        result = qa_chain.invoke(q)
        contexts = [d.page_content for d in result["source_documents"]]
        samples.append(
            SingleTurnSample(
                user_input=q,
                retrieved_contexts=contexts,
                response=result["result"],
                reference=truth,  # ground-truth answer
            )
        )

    dataset = EvaluationDataset(samples=samples)
    # Eval
    metrics = [context_precision, context_recall,
               answer_relevancy, faithfulness]

    scores = evaluate(dataset=dataset, metrics=metrics, llm=evaluator_llm, embeddings=evaluator_emb)

    print("\n── RAGAS EVALUATION ─────────────")
    for m in metrics:
        print(f"{m.name:17s}: {scores[m.name]:.3f}")
    print("──────────────────────────────────")

if __name__ == '__main__':
    main()