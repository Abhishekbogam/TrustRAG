import os
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS


# Configuration
FAISS_PATH = "data/faiss_index"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 200
TOP_K = 5


# Embeddings
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


# Text Splitter
def get_text_splitter():
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )


# FAISS Lifecycle
def load_faiss_if_exists(embeddings):
    os.makedirs(FAISS_PATH, exist_ok=True)
    index_file = os.path.join(FAISS_PATH, "index.faiss")

    if os.path.exists(index_file):
        return FAISS.load_local(
            FAISS_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )

    return None

# Ingestion
def ingest_documents(documents):
    embeddings = get_embeddings()
    splitter = get_text_splitter()
    docs = splitter.split_documents(documents)

    if not docs:
        raise ValueError("No valid document content to ingest.")

    vectorstore = load_faiss_if_exists(embeddings)

    if vectorstore is None:
        # First ingestion
        vectorstore = FAISS.from_documents(docs, embeddings)
    else:
        # Append
        vectorstore.add_documents(docs)

    vectorstore.save_local(FAISS_PATH)


# Retriever
def get_retriever():
    embeddings = get_embeddings()
    vectorstore = load_faiss_if_exists(embeddings)

    if vectorstore is None:
        raise RuntimeError("FAISS index not initialized. Upload documents first.")

    return vectorstore.as_retriever(
        search_kwargs={"k": TOP_K}
    )
