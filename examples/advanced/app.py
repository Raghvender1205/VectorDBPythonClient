import os
import streamlit as st
import logging
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

# Load environment variables
load_dotenv(find_dotenv("../.env"))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "collection_name" not in st.session_state:
    st.session_state.collection_name = ""

def initialize_client():
    """Initialize VectorDB client"""
    server_url = "http://127.0.0.1:8444"
    return VectorDBClient(server_url=server_url)

def get_embeddings():
    """Get embedding model"""
    return OpenAIEmbeddings(
        base_url=os.getenv("EMBEDDING_URL"),
        api_key=os.getenv("EMBEDDING_API_KEY"),
        model=os.getenv("EMBEDDING_MODEL_NAME"),
    )

def get_llm():
    """Get LLM model"""
    return ChatOpenAI(
        base_url=os.getenv("LLM_BASE_URL"),
        api_key=os.getenv("LLM_API_KEY"),
        model=os.getenv("LLM_MODEL_NAME"),
    )

def process_document(file, collection_name, client):
    """Process uploaded document and add to collection"""
    try:
        # Save uploaded file temporarily
        with open("temp_doc.pdf", "wb") as f:
            f.write(file.getvalue())
        
        # Load and chunk document
        loader = PyPDFLoader("temp_doc.pdf")
        docs = loader.load()
        
        # Prepare texts and metadatas
        texts = []
        metadatas = []
        for idx, doc in enumerate(docs, start=1):
            text = doc.page_content.strip()
            if text:
                metadata = {"category": "pdf_document", "page_number": idx}
                texts.append(text)
                metadatas.append(metadata)
        
        if not texts:
            st.error("No text content found in the document.")
            return False
        
        # Get embedding model
        embedding_model = get_embeddings()
        
        # Add to vector store
        vectordb_store = VectorDBVectorStore.from_texts(
            texts=texts,
            embedding=embedding_model,
            metadatas=metadatas,
            client=client,
            collection_name=collection_name,
            additional_metadata={"source": "PDF Document"},
        )
        
        # Clean up temporary file
        os.remove("temp_doc.pdf")
        return True
        
    except Exception as e:
        st.error(f"Error processing document: {str(e)}")
        return False

def initialize_qa_chain(collection_name, client):
    """Initialize QA chain with the specified collection"""
    try:
        embedding_model = get_embeddings()
        llm = get_llm()
        
        # Create vector store
        vectordb_store = VectorDBVectorStore(
            client=client,
            collection_name=collection_name,
            embedding_model=embedding_model,
        )
        
        # Create QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectordb_store.as_retriever(),
            return_source_documents=True,
        )
        
        return qa_chain
    except Exception as e:
        st.error(f"Error initializing QA chain: {str(e)}")
        return None

def main():
    st.set_page_config(page_title="Document Chat", page_icon="🤖", layout="wide")
    
    st.title("📚 Document Chat Assistant")
    
    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        
        # Mode selection
        mode = st.radio(
            "Select Mode",
            ["Chat Only", "Ingest & Chat"],
            help="Choose whether to chat with existing documents or ingest new ones"
        )
        
        # Collection management
        st.subheader("Collection Management")
        collection_option = st.radio(
            "Collection",
            ["Use Existing", "Create New"],
            help="Choose to use an existing collection or create a new one"
        )
        
        if collection_option == "Use Existing":
            client = initialize_client()
            try:
                collections = client.list_collections()
                collection_names = [col.name for col in collections]
                if collection_names:
                    st.session_state.collection_name = st.selectbox(
                        "Select Collection",
                        collection_names
                    )
                else:
                    st.warning("No collections found. Please create a new collection.")
                    collection_option = "Create New"
            except Exception as e:
                st.error(f"Error fetching collections: {str(e)}")
                return
        
        if collection_option == "Create New":
            st.session_state.collection_name = st.text_input(
                "New Collection Name",
                placeholder="Enter collection name"
            )
        
        # Document upload for Ingest mode
        if mode == "Ingest & Chat":
            st.subheader("Document Upload")
            uploaded_file = st.file_uploader(
                "Upload PDF Document",
                type=["pdf"],
                help="Upload a PDF document to ingest into the collection"
            )
            
            if uploaded_file and st.session_state.collection_name:
                if st.button("Process Document"):
                    with st.spinner("Processing document..."):
                        client = initialize_client()
                        if process_document(uploaded_file, st.session_state.collection_name, client):
                            st.success("Document processed successfully!")
                            # Initialize QA chain after successful ingestion
                            st.session_state.qa_chain = initialize_qa_chain(
                                st.session_state.collection_name,
                                client
                            )
    
    # Main chat interface
    if st.session_state.collection_name:
        # Initialize QA chain if not already done
        if st.session_state.qa_chain is None:
            client = initialize_client()
            st.session_state.qa_chain = initialize_qa_chain(
                st.session_state.collection_name,
                client
            )
        
        # Chat interface
        st.subheader("Chat")
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
                if "sources" in message:
                    with st.expander("View Sources"):
                        for source in message["sources"]:
                            st.write(f"Page {source['page_number']}: {source['text'][:200]}...")
        
        # Chat input
        if prompt := st.chat_input("Ask a question about your documents"):
            if st.session_state.qa_chain is None:
                st.error("Please wait for the QA chain to initialize or check your collection settings.")
                return
            
            # Add user message to chat
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)
            
            # Get response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        response = st.session_state.qa_chain.invoke(prompt)
                        st.write(response["result"])
                        
                        # Add assistant message to chat
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": response["result"],
                            "sources": [
                                {
                                    "page_number": doc.metadata.get("page_number"),
                                    "text": doc.page_content
                                }
                                for doc in response["source_documents"]
                            ]
                        })
                        
                        # Display sources in expander
                        with st.expander("View Sources"):
                            for doc in response["source_documents"]:
                                st.write(f"Page {doc.metadata.get('page_number')}: {doc.page_content[:200]}...")
                    
                    except Exception as e:
                        st.error(f"Error getting response: {str(e)}")
    else:
        st.info("Please select or create a collection to start chatting.")

if __name__ == "__main__":
    main()