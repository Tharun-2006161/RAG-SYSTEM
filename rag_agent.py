"""
Self-Healing RAG Agent
A LangGraph-powered agent that retrieves documents, generates answers,
grades them for groundedness, and retries with rewritten questions if needed.
"""

import os
from typing import TypedDict
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
import chromadb
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

load_dotenv()


# ── Step 2: Define the Agent State ───────────────────────────────────────────

class AgentState(TypedDict):
    question: str               # the user's original question
    rewritten_question: str     # a rephrased version for retry
    documents: list[str]        # chunks retrieved from ChromaDB
    answer: str                 # the generated answer
    grade: str                  # "pass" or "fail"
    retry_count: int            # how many retries have happened


# ── Step 3: Initialize the LLM and Vector Store ─────────────────────────────

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=os.environ.get("GROQ_API_KEY")
)

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(
    name="knowledge_base",
    metadata={"hnsw:space": "cosine"}
)


# ── Step 4: The Retrieve Node ────────────────────────────────────────────────

def retrieve(state: AgentState) -> AgentState:
    """Search ChromaDB for documents relevant to the current question."""
    question = state.get("rewritten_question") or state["question"]

    print(f"\n[RETRIEVE] Searching for: {question}")

    try:
        results = collection.query(
            query_texts=[question],
            n_results=3
        )
        documents = results["documents"][0] if results["documents"] else []
    except Exception as e:
        print(f"[RETRIEVE] Error: {e}")
        documents = []

    print(f"[RETRIEVE] Found {len(documents)} documents.")
    return {**state, "documents": documents}


# ── Step 5: The Generate Node ────────────────────────────────────────────────

def generate(state: AgentState) -> AgentState:
    """Generate an answer using the LLM based on retrieved documents."""
    question = state.get("rewritten_question") or state["question"]
    documents = state.get("documents", [])

    if not documents:
        return {**state, "answer": "No relevant documents were found."}

    context = "\n\n".join([f"Document {i+1}:\n{doc}"
                           for i, doc in enumerate(documents)])

    system_prompt = """You are a helpful assistant that answers questions
based strictly on the provided documents. If the documents do not contain
enough information to answer the question, say so honestly.
Do not use any knowledge outside of the provided documents."""

    user_prompt = f"""Documents:
{context}

Question: {question}

Answer based only on the documents above:"""

    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])
        answer = response.content
        print(f"[GENERATE] Answer generated ({len(answer)} chars).")
    except Exception as error:
        print(f"[GENERATE] LLM call failed: {error}")
        answer = "An error occurred while generating the answer."

    return {**state, "answer": answer}


# ── Step 6: The Grade Node ───────────────────────────────────────────────────

def grade_answer(state: AgentState) -> AgentState:
    """Grade whether the answer is grounded in the retrieved documents."""
    answer = state["answer"]
    documents = state.get("documents", [])
    question = state.get("rewritten_question") or state["question"]

    context = "\n".join(documents)

    grade_prompt = f"""You are a grading assistant. Your job is to determine
if an answer is grounded in and supported by the provided documents.

Documents:
{context}

Question: {question}

Answer: {answer}

Is this answer grounded in the documents? Reply with ONLY "pass" or "fail"."""

    try:
        response = llm.invoke([HumanMessage(content=grade_prompt)])
        grade = response.content.strip().lower()
        if "pass" in grade:
            grade = "pass"
        else:
            grade = "fail"
    except Exception as e:
        print(f"[GRADE] Error: {e}")
        grade = "fail"

    print(f"[GRADE] Result: {grade}")
    return {**state, "grade": grade}


# ── Step 7: The Rewrite Node ─────────────────────────────────────────────────

def rewrite_question(state: AgentState) -> AgentState:
    """Rephrase the question for a better retrieval attempt."""
    question = state.get("rewritten_question") or state["question"]
    retry_count = state.get("retry_count", 0)

    rewrite_prompt = f"""You are a question rewriter. Your goal is to rephrase
the following question to get better search results from a document database.

Original question: {question}

Rewrite the question to be clearer and more specific.
Return ONLY the rewritten question, nothing else."""

    try:
        response = llm.invoke([HumanMessage(content=rewrite_prompt)])
        new_question = response.content.strip()
    except Exception as e:
        print(f"[REWRITE] Error: {e}")
        new_question = question

    print(f"[REWRITE] New question: {new_question}")
    return {**state, "rewritten_question": new_question, "retry_count": retry_count + 1}


# ── Step 8: The Give Up Node ─────────────────────────────────────────────────

def give_up(state: AgentState) -> AgentState:
    """Return an honest 'I don't know' when max retries are exceeded."""
    print("[GIVE UP] Max retries exceeded. Returning 'I don't know'.")
    return {
        **state,
        "answer": "I don't know. The system could not find a well-supported "
                   "answer after multiple attempts."
    }


# ── Step 9: Routing Logic ────────────────────────────────────────────────────

def route_after_grade(state: AgentState) -> str:
    """Decide next step after grading the answer."""
    if state["grade"] == "pass":
        return END
    if state.get("retry_count", 0) >= 2:
        return "give_up"
    return "rewrite_question"


# ── Step 10: Build the Graph ─────────────────────────────────────────────────

workflow = StateGraph(AgentState)

# Add all nodes
workflow.add_node("retrieve", retrieve)
workflow.add_node("generate", generate)
workflow.add_node("grade_answer", grade_answer)
workflow.add_node("rewrite_question", rewrite_question)
workflow.add_node("give_up", give_up)

# Set the entry point
workflow.set_entry_point("retrieve")

# Add edges (the flow)
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", "grade_answer")

# Conditional branching after grading
workflow.add_conditional_edges(
    "grade_answer",
    route_after_grade,
    {
        END: END,
        "rewrite_question": "rewrite_question",
        "give_up": "give_up",
    }
)

# Rewrite loops back to retrieve
workflow.add_edge("rewrite_question", "retrieve")

# Give up ends the graph
workflow.add_edge("give_up", END)

# Compile the graph
app = workflow.compile()


# ── Step 11: Main Runner ─────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 50)
    print("  Self-Healing RAG Agent")
    print("  Type 'quit' to exit")
    print("=" * 50)

    while True:
        question = input("\nAsk a question: ").strip()

        if question.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        if not question:
            continue

        result = app.invoke({
            "question": question,
            "rewritten_question": "",
            "documents": [],
            "answer": "",
            "grade": "",
            "retry_count": 0,
        })

        print(f"\n{'=' * 50}")
        print(f"Final Answer: {result['answer']}")
        print(f"{'=' * 50}")
