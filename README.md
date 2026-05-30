# 🧠 Self-Healing RAG System

A production-ready Retrieval-Augmented Generation (RAG) chatbot that answers questions from your own documents, validates its responses, and automatically retries when answers are not grounded in the source material.

## 🚀 Overview

Traditional RAG systems retrieve relevant documents and generate answers from them. However, they can still hallucinate and provide incorrect information.

The Self-Healing RAG System solves this problem by introducing a verification layer that grades generated answers before returning them to the user.

### Key Idea

Instead of:

Question → Retrieve → Generate → Answer

This system performs:

Question → Retrieve → Generate → Grade → Retry if Needed → Answer

If the answer cannot be verified after multiple attempts, the system honestly responds:

> "I don't know based on the provided documents."

---

## ✨ Features

- 📚 Document-based Question Answering
- 🔍 Semantic Search using ChromaDB
- 🤖 Answer Generation using Groq Llama 3.3 70B
- ✅ Answer Verification (Groundedness Check)
- 🔄 Automatic Question Rewriting & Retry
- 🚫 Hallucination Reduction
- 🏗️ LangGraph Workflow Architecture
- 💻 Runs Locally with Free Embedding Models
- 📈 Easy to Extend with New Documents

---

## 🏗️ Architecture

```text
User Question
      ↓
Retrieve Documents
      ↓
Generate Answer
      ↓
Grade Answer
   /          \
 PASS         FAIL
  |             |
Return      Rewrite Question
Answer           |
                 ↓
          Retrieve Again
                 ↓
            Retry Limit?
            /        \
          No         Yes
          |           |
        Retry     I Don't Know
```

---

## 🛠️ Tech Stack

| Technology | Purpose |
|------------|----------|
| Python 3.10+ | Core Programming Language |
| ChromaDB | Vector Database |
| Groq API | Llama 3.3 70B Inference |
| LangChain | LLM & Retrieval Framework |
| LangGraph | Workflow Orchestration |
| python-dotenv | Environment Variables |
| uv | Package Management |
| all-MiniLM-L6-v2 | Embedding Model |

---

## 📂 Project Structure

```text
RAG_SYSTEM/
│
├── .env
├── .gitignore
├── pyproject.toml
├── uv.lock
│
├── docs/
│   ├── python_basics.txt
│   └── machine_learning.txt
│
├── ingest.py
├── rag_agent.py
│
└── chroma_db/
```

---

## 📁 File Descriptions

### ingest.py

Responsible for:

- Reading all documents from `docs/`
- Splitting documents into chunks
- Generating embeddings
- Storing vectors in ChromaDB

### rag_agent.py

Main AI agent containing:

- Retrieval Node
- Generation Node
- Grading Node
- Rewrite Node
- Give-Up Node

---

## 🔄 Workflow

### Step 1: User Asks a Question

Example:

```text
What is Python?
```

---

### Step 2: Retrieve Relevant Documents

The system searches ChromaDB and retrieves the most relevant chunks.

---

### Step 3: Generate an Answer

Groq Llama 3.3 generates an answer using:

- User Question
- Retrieved Context

---

### Step 4: Grade the Answer

The system checks:

- Is the answer supported by the retrieved documents?
- Is the answer hallucinated?

Possible results:

```text
PASS
```

or

```text
FAIL
```

---

### Step 5: Self-Healing

If FAIL:

1. Rewrite Question
2. Search Again
3. Generate New Answer
4. Re-grade

Maximum retries: 2

---

### Step 6: Final Output

If answer is grounded:

```text
Return Answer
```

Otherwise:

```text
I don't know based on the provided documents.
```

---

## ⚙️ Installation

### Clone Repository

```bash
git clone <repository-url>
cd RAG_SYSTEM
```

### Install Dependencies

Using uv:

```bash
uv sync
```

Or using pip:

```bash
pip install -r requirements.txt
```

---

## 🔑 Environment Variables

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_api_key
```

Get a free API key from Groq.

---

## 📥 Ingest Documents

Place your documents inside:

```text
docs/
```

Example:

```text
docs/
├── company_policy.txt
├── employee_handbook.txt
└── product_manual.txt
```

Run ingestion:

```bash
uv run ingest.py
```

This creates vector embeddings and stores them in ChromaDB.

---

## ▶️ Run the Chatbot

```bash
uv run rag_agent.py
```

Example:

```text
Ask a question:
> What is Python?
```

---

## 📊 Example Output

### Question

```text
What is Python?
```

### Response

```text
Python is a high-level interpreted programming language
created by Guido van Rossum in 1991.
```

---

### Question

```text
What is the capital of France?
```

### Response

```text
The provided documents do not contain enough information
to answer this question.
```

---

## 🎯 Benefits

### Reduced Hallucinations

Answers are validated before being shown.

### Reliable Knowledge Base

Ideal for:

- Company Policies
- HR Portals
- University Information Systems
- Product Documentation
- Legal Documents
- Internal Knowledge Bases

### Cost Effective

- Free Groq API
- Local Embeddings
- No GPU Required

---

## 🔮 Future Improvements

- PDF Support
- DOCX Support
- Multi-Document Collections
- Web Interface (Streamlit/React)
- Conversation Memory
- Source Citations
- User Authentication
- Feedback-Based Learning

---

## 🧪 Example Use Cases

### University Assistant

Answer admission and placement questions from university documents.

### HR Assistant

Answer employee policy questions.

### Product Support Bot

Answer questions from manuals and documentation.

### Legal Document Assistant

Provide grounded answers from contracts and agreements.

---

## 📈 Why This Project Matters

This project demonstrates practical skills in:

- Retrieval-Augmented Generation (RAG)
- Vector Databases
- Prompt Engineering
- LLM Evaluation
- LangChain
- LangGraph
- AI Workflow Design
- Production AI Systems

It is an excellent portfolio project for AI, ML, and Software Engineering interviews.

---

## 📜 License

MIT License

---

## 👨‍💻 Author

Built with ❤️ using Python, LangChain, LangGraph, ChromaDB, and Groq.
