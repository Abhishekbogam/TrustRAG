
# TrustRAG

A document-grounded, memory-aware, source-attributed RAG system built with FastAPI, Streamlit, FAISS, and LLaMA-3 (Groq).


## Overview

Large Language Models (LLMs) often face issues such as hallucinations, outdated knowledge, and limited traceability when working with complex technical documents.

TrustRAG is a Retrieval-Augmented Generation (RAG) system that grounds responses in user-provided documents using conversational memory, domain-aware controls, and source attribution. It is optimized for technical domains like telecommunications and works effectively with large standards documents such as 3GPP specifications, while remaining configurable for other domains and use cases.
## Key features

1. Document-Grounded Answers

Answers are generated using only retrieved document content.

2. Conversation Memory

Maintains short-term conversational context across turns.

3. Semantic Search with FAISS

High-performance similarity search over embedded document chunks.

4. Smart Chunking Strategy

Overlapping chunks preserve semantic continuity.

5. Domain Gating

Detects out-of-scope questions and avoids hallucinated answers.

6. Source Attribution

Displays only document filenames (not paths or hallucinated citations).

7. Typing Animation (UX)

ChatGPT-like typing effect for summaries and bullet points.

8. Structured Responses

Supports paragraphs, bullet points, and tables consistently.
##  Architecture

![System Architecture](https://github.com/Abhishekbogam/Dummy/raw/1dad28efcf3dd4a94de1b27284de9efc3decacfa/RAG_APPLication.png)
## Techstack

--- 
| Layer        | Technology                              |
|--------------|------------------------------------------|
| Frontend     | Streamlit                                |
| Backend API  | FastAPI                                  |
| Embeddings   | MiniLM (Sentence-Transformers)           |
| Vector Store | FAISS                                    |
| LLM          | LLaMA-3.3-70B via Groq                   |
| Memory       | Sliding Window (Last N Turns)            |


## Core Workflow

## Document Ingestion Pipeline

 1.	User uploads PDF/TXT files via Streamlit
 2.	Backend saves files locally
 3.	Documents are loaded using LangChain loaders
 4.	Text is split into overlapping chunks
 5.	Each chunk is embedded using MiniLM
 6.	Embeddings are stored in FAISS
 7.	Filename is stored as metadata for traceability
 .

Why chunking matters
Chunk overlap ensures that technical definitions and jargon remain intact across boundaries.



![Document Ingestion Pipeline](https://github.com/Abhishekbogam/Dummy/raw/a5955f7ece1223f24b3bb646d441731039081fbe/Document%20Ingestion%20Pipeline.png)

## Query & Answer Pipeline

1.	User submits a query
2.	Query embedding is generated
3.	FAISS retrieves top-K relevant chunks
4.	Domain gating checks relevance score
5.	Conversation memory is appended
6.	Context is passed to LLaMA-3
7.	LLM returns structured JSON
8.	Backend attaches verified sources
9.	Streamlit renders formatted output
...
![Query Pipeline](https://github.com/Abhishekbogam/Dummy/raw/d52ccfe789e289882c50974363db903cab3f8024/Query%20Pipeline.png)

## Conversation Memory Strategy

1. Maintain Recent Context

Uses a sliding window of the last N user–assistant turns

2. Context-Aware Memory Usage

Memory is applied only when the current question refers to previous context

3. Prevent Context Overload

Avoids prompt bloating and context pollution by limiting memory usage

Example Interaction

User: Explain network topology

Bot: Network topology refers to...

User: Explain its advantages

Bot: Advantages are ...

![Conversation Memory Strategy](https://github.com/Abhishekbogam/Dummy/raw/c4770ba955c6e5bb30848cf547b64a4afc7c4040/conversation%20memory1.png)


## Trust & Hallucination Control

    Domain Gating Logic

    •	In-Domain → Answer strictly from documents

    •	Weak Coverage → Hybrid response (document + limited general knowledge)

    •	Out-of-Domain → Safe refusal

This ensures:

    •	No fabricated technical facts
    •	No fake citations
    •	Predictable behavior


## Source Attribution

•	Sources are extracted from document metadata

•	Only filenames are shown (e.g., 0568495.pdf)

•	LLM is never allowed to generate sources

•	Backend controls all attribution

## Sample Outputs

### Output 1
![Output 1](https://github.com/Abhishekbogam/Dummy/raw/7f69cc151971b8f4ff2f6e3fa9dfeb31bc4cc0d5/output_1.png)

### Output 2
![Output 2](https://github.com/Abhishekbogam/Dummy/raw/7f69cc151971b8f4ff2f6e3fa9dfeb31bc4cc0d5/output_2.png)

## Prerequisites



Python 3.9 – 3.11 (recommended: 3.10)

Git

Internet access (for Groq API and model downloads)

### Required Accounts / Keys

Groq API Key (for LLaMA-3 inference) 

https://console.groq.com/keys

(Optional) Hugging Face access for embedding models (auto-downloaded)

### Installation

1️⃣ Clone the Repository

    git clone https://github.com//TrustRAG.git
    cd TrustRAG

2️⃣ Create a Virtual Environment

    python -m venv venv


Activate it:

Linux / macOS

    source venv/bin/activate


Windows

    venv\Scripts\activate

3️⃣ Install Dependencies

    pip install -r requirements.txt


4️⃣ Environment Variables

copy paste the API key .env file

    GROQ_API_KEY=your_groq_api_key_here

5️⃣ Run the Backend (FastAPI)

    uvicorn main:app --reload


6️⃣ Run the Frontend (Streamlit)

In a new terminal:

    streamlit run app.py


Frontend will open at:

    http://localhost:8501

## Conclusion

TrustRAG demonstrates how RAG systems can be built responsibly, transparently, and at production quality — especially for technical domains where correctness and trust matter.

This project is not just a chatbot, but a retrieval-first, source-aware AI system.
