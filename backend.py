import os
import json
from typing import List
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain_groq import ChatGroq
from langchain.document_loaders import PyPDFLoader, TextLoader
from rag_utils import ingest_documents, get_retriever
from sentence_transformers import CrossEncoder


# API keys
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found. Please set it in the .env file.")

_reranker = None

def get_reranker():
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _reranker


app = FastAPI(title="RAG GenAI Backend", version="1.0.0")

# Upload API
@app.post("/upload")
async def upload_documents(files: List[UploadFile] = File(...)):
    documents = []

    for file in files:
        os.makedirs("data/uploads", exist_ok=True)
        file_path = f"data/uploads/{file.filename}"

        with open(file_path, "wb") as f:
            f.write(await file.read())

        if file.filename.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        else:
            loader = TextLoader(file_path)

        # documents.extend(loader.load())
        loaded_docs = loader.load()

        for doc in loaded_docs:
            doc.metadata["source"] = file.filename

        documents.extend(loaded_docs)


    ingest_documents(documents)
    return JSONResponse({"status": "success", "files_ingested": len(files)})


# Prompt (JSON Structured Output)
PROMPT = PromptTemplate.from_template(
    """
You are a technical assistant using retrieval-augmented generation.

Answer the question primarily using the provided context.

If the context is insufficient AND the question is within the same technical domain:
- You MAY use limited general technical knowledge.
- Clearly state this in the summary.
- Do NOT fabricate document-specific facts.

Return the answer STRICTLY in the following JSON format.
Do NOT add any text outside JSON.

JSON SCHEMA:
{{
  "summary": "string",
  "bullet_points": ["string"],
  "table": {{
    "headers": ["string"],
    "rows": [["string"]]
  }}
}}

RULES:
- summary is mandatory.
- bullet_points must be empty list if not applicable.
- table.headers and table.rows must be empty if not applicable.
- Do NOT mention sources, references, authors, books, or citations.
- Do NOT use markdown.

Use the conversation context ONLY if the question refers to previous answers.

CONTEXT:
{context}

QUESTION:
{question}
"""
)


# LLM (NO STREAMING — IMPORTANT)
llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model_name="llama-3.3-70b-versatile",
    streaming=False,
    temperature=0.2,
    max_tokens=2048
)

# Query API
# ---------------- Domain & Retrieval Config ----------------
TOP_K = 5
STRICT_THRESHOLD = 1.2
DOMAIN_THRESHOLD = 3.0


@app.post("/query")
async def query_rag(payload: dict):
    question = payload.get("question")
    memory = payload.get("memory", "")

    retriever = get_retriever()
    

    # ---------------- Retrieval with scores ----------------
    docs_with_scores = retriever.vectorstore.similarity_search_with_score(
        question, k=TOP_K
    )

    if not docs_with_scores:
        return JSONResponse(content={
            "summary": (
                "I can only answer questions related to the uploaded documents. "
                "The current question does not appear to be covered by them."
            ),
            "bullet_points": [],
            "table": {"headers": [], "rows": []},
            "sources": []
        })

    # docs = [doc for doc, score in docs_with_scores]

    docs = [doc for doc, score in docs_with_scores]

    # DEBUG: FAISS results BEFORE reranking
    # print("\n--- FAISS TOP-K ---")
    # for i, d in enumerate(docs):
    #     print(f"{i+1}. {d.metadata.get('source')} | {d.page_content[:80]}")

    # ---------------- RERANKING ----------------
    reranker = get_reranker()

    pairs = [(question, doc.page_content) for doc in docs]
    rerank_scores = reranker.predict(pairs)

    reranked = sorted(
        zip(docs, rerank_scores),
        key=lambda x: x[1],
        reverse=True
    )

    # Keep top 3 after reranking
    docs = [doc for doc, _ in reranked[:3]]

    # DEBUG: Results AFTER reranking
    # print("\n--- RERANKED TOP ---")
    # for i, d in enumerate(docs):
    #     print(f"{i+1}. {d.metadata.get('source')} | {d.page_content[:80]}")


    # ---------------- Reranking ----------------
    reranker = get_reranker()

    pairs = [
        (question, doc.page_content)
        for doc in docs
    ]

    scores = reranker.predict(pairs)

    reranked = sorted(
        zip(docs, scores),
        key=lambda x: x[1],
        reverse=True
    )

    # Keep top 3 after reranking
    docs = [doc for doc, _ in reranked[:3]]



    # for d in docs:
        # print("SOURCE METADATA:", d.metadata.get("source"))

    scores = [score for doc, score in docs_with_scores]

    # ---------------- Domain gating ----------------
    
    min_score = min(scores)

    #  Completely out of domain
    if min_score > DOMAIN_THRESHOLD:
        return JSONResponse(content={
            "summary": (
                "I can only answer questions related to the uploaded documents. "
                "This question appears to be outside the document scope."
            ),
            "bullet_points": [],
            "table": {"headers": [], "rows": []},
            "sources": []
        })

    #  Weakly covered but same domain
    hybrid_mode = min_score > STRICT_THRESHOLD


    # ---------------- Sources ----------------
    sources = sorted({
        os.path.basename(d.metadata.get("source", "Unknown"))
        for d in docs
    })

    # ---------------- Context ----------------
    doc_context = "\n\n".join(d.page_content for d in docs)

    if hybrid_mode:
        context = f"""
    Conversation so far:
    {memory}

    Relevant documents (may be partial):
    {doc_context}
    """
    else:
        context = f"""
    Conversation so far:
    {memory}

    Relevant documents:
    {doc_context}
    """


    # ---------------- LLM Chain ----------------
    chain = (
        {
            "context": lambda _: context,
            "question": RunnablePassthrough()
        }
        | PROMPT
        | llm
    )

    raw_output = chain.invoke(question).content

    try:
        parsed = json.loads(raw_output)
    except json.JSONDecodeError:
        parsed = {
            "summary": raw_output,
            "bullet_points": [],
            "table": {"headers": [], "rows": []}
        }

    # ---------------- Safe source handling ----------------
    if hybrid_mode:
        parsed["sources"] = []
    else:
        parsed["sources"] = sources
    # if "sources" not in parsed:
    #     parsed["sources"] = sources

    return JSONResponse(content=parsed)

