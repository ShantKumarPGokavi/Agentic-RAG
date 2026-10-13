import atexit
import html
import os
import re
from pathlib import Path
from typing import TypedDict

import requests
from dotenv import load_dotenv
from ddgs import DDGS
from ddgs.exceptions import DDGSException
from langgraph.graph import END, StateGraph
from openai import OpenAI
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer


load_dotenv()

COLLECTION_NAME = "company_knowledge"
QDRANT_PATH = Path(__file__).parent / "qdrant_data"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

embedding_model = None
qdrant_client = None


class AssistantState(TypedDict, total=False):
    question: str
    documents: list[dict]
    grade: str
    web_results: list[dict]
    source: str
    answer: str


def get_api_key():
    return (
        os.getenv("LLM_API_KEY")
        or os.getenv("GROQ_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or os.getenv("GEMINI_API_KEY")
    )


def get_resources():
    global embedding_model, qdrant_client

    if qdrant_client is None:
        qdrant_client = QdrantClient(path=str(QDRANT_PATH))
    if not qdrant_client.collection_exists(COLLECTION_NAME):
        raise RuntimeError("Knowledge base is empty. Run `python ingest.py` first.")
    if embedding_model is None:
        embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    return embedding_model, qdrant_client


def close_resources():
    global qdrant_client

    if qdrant_client is not None:
        qdrant_client.close()
        qdrant_client = None


atexit.register(close_resources)


def call_llm(system_prompt, user_prompt):
    api_key = get_api_key()
    if not api_key or api_key == "your_api_key_here":
        raise RuntimeError(
            "Set LLM_API_KEY in your .env file before asking a question."
        )

    client = OpenAI(
        api_key=api_key,
        base_url=os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1"),
    )
    response = client.chat.completions.create(
        model=os.getenv("LLM_MODEL", "openai/gpt-oss-20b"),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
    )
    return response.choices[0].message.content or ""


def retrieve_node(state):
    model, client = get_resources()
    vector = model.encode(state["question"]).tolist()
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=vector,
        limit=4,
        with_payload=True,
    ).points

    documents = [
        {
            "title": result.payload["title"],
            "text": result.payload["text"],
        }
        for result in results
    ]
    return {"documents": documents, "source": "Local Knowledge Base"}


def grade_node(state):
    context = "\n\n".join(
        f"{document['title']}: {document['text']}"
        for document in state["documents"]
    )
    grade = call_llm(
        "Decide whether the supplied documents contain enough information to answer "
        "the question. Reply with only yes or no.",
        f"Question: {state['question']}\n\nDocuments:\n{context}",
    )
    return {"grade": "yes" if grade.strip().lower().startswith("yes") else "no"}


def choose_next_node(state):
    if state["grade"] == "yes":
        return "generate_node"
    return "web_search_node"


def web_search_node(state):
    try:
        results = list(DDGS().text(state["question"], max_results=5))
    except DDGSException:
        results = []
    web_results = [
        {
            "title": result.get("title", "Search result"),
            "text": result.get("body", ""),
            "url": result.get("href", ""),
        }
        for result in results
        if result.get("body")
    ]
    if not web_results:
        response = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "list": "search",
                "srsearch": state["question"],
                "format": "json",
                "srlimit": 5,
            },
            headers={"User-Agent": "SelfCorrectingRAGAssistant/1.0"},
            timeout=15,
        )
        response.raise_for_status()
        pages = response.json().get("query", {}).get("search", [])
        web_results = [
            {
                "title": page["title"],
                "text": html.unescape(
                    re.sub(r"<[^>]+>", "", page.get("snippet", ""))
                ),
                "url": f"https://en.wikipedia.org/?curid={page['pageid']}",
            }
            for page in pages
            if page.get("snippet")
        ]
    return {"web_results": web_results, "source": "Live Web Search"}


def generate_node(state):
    if state["source"] == "Local Knowledge Base":
        evidence = "\n\n".join(
            f"{document['title']}: {document['text']}"
            for document in state["documents"]
        )
    else:
        evidence = "\n\n".join(
            f"{result['title']}: {result['text']} ({result['url']})"
            for result in state.get("web_results", [])
        )
        if not evidence:
            evidence = "No useful web search results were found."

    answer = call_llm(
        "Answer the question using only the supplied evidence. Be clear when the "
        "evidence does not contain the answer. Do not invent facts. Include relevant "
        "web URLs as markdown links when they are provided.",
        f"Question: {state['question']}\n\nEvidence:\n{evidence}",
    )
    return {"answer": f"{answer.strip()}\n\nSource: {state['source']}."}


workflow = StateGraph(AssistantState)
workflow.add_node("retrieve_node", retrieve_node)
workflow.add_node("grade_node", grade_node)
workflow.add_node("web_search_node", web_search_node)
workflow.add_node("generate_node", generate_node)
workflow.set_entry_point("retrieve_node")
workflow.add_edge("retrieve_node", "grade_node")
workflow.add_conditional_edges(
    "grade_node",
    choose_next_node,
    {
        "web_search_node": "web_search_node",
        "generate_node": "generate_node",
    },
)
workflow.add_edge("web_search_node", "generate_node")
workflow.add_edge("generate_node", END)
assistant_graph = workflow.compile()


def answer_question(question):
    result = assistant_graph.invoke({"question": question})
    if result["source"] == "Local Knowledge Base":
        sources = [document["title"] for document in result["documents"]]
    else:
        sources = [
            {"title": item["title"], "url": item["url"]}
            for item in result.get("web_results", [])
        ]
    return {
        "answer": result["answer"],
        "source": result["source"],
        "sources": sources,
    }
