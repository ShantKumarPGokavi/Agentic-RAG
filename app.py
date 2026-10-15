import os

import requests
import streamlit as st


st.set_page_config(
    page_title="RAG & Web Fallback Assistant",
    page_icon="🔎",
    layout="centered",
)

st.title("Self-Correcting RAG Assistant")
st.caption(
    "Answers from the local company knowledge base first, then searches the web "
    "when the documents are not enough. Start the FastAPI service before asking."
)

api_url = os.getenv("API_URL", "http://127.0.0.1:8000")

with st.form("ask_form"):
    question = st.text_area(
        "Your question",
        placeholder="For example: Can I paste customer data into a public AI tool?",
        max_chars=1000,
    )
    submitted = st.form_submit_button("Ask")

if submitted:
    if not question.strip():
        st.warning("Enter a question first.")
    else:
        result = None
        try:
            with st.spinner("Looking for a grounded answer..."):
                response = requests.post(
                    f"{api_url}/ask",
                    json={"question": question.strip()},
                    timeout=120,
                )
            if response.ok:
                try:
                    result = response.json()
                except ValueError:
                    st.error("The API returned an invalid JSON response.")
                if result is None:
                    st.error("The API returned an empty response.")
            else:
                try:
                    error_body = response.json()
                except ValueError:
                    error_body = {}
                detail = (
                    error_body.get("detail")
                    if isinstance(error_body, dict)
                    else None
                )
                st.error(detail or response.text or "The API request failed.")
        except requests.RequestException as error:
            st.error(f"Could not reach the FastAPI service: {error}")

        if result is not None:
            if not isinstance(result, dict):
                st.error("The API returned an unexpected response.")
            elif not isinstance(result.get("answer"), str):
                st.error("The API response did not include an answer.")
            else:
                source = result.get("source")
                sources = result.get("sources", [])
                if source == "Local Knowledge Base" and isinstance(sources, list):
                    st.success("Path taken: Local Docs")
                    document_titles = [item for item in sources if isinstance(item, str)]
                    st.write("Documents used: " + ", ".join(document_titles))
                elif source == "Live Web Search" and isinstance(sources, list):
                    st.info("Path taken: Web Search")
                    if sources:
                        st.write("Search results:")
                        for item in sources:
                            if isinstance(item, dict) and item.get("url"):
                                title = item.get("title", "Search result")
                                st.markdown(f"- [{title}]({item['url']})")
                            elif isinstance(item, dict):
                                st.write(f"- {item.get('title', 'Search result')}")
                else:
                    st.error("The API response did not include a valid answer source.")
                st.subheader("Answer")
                st.markdown(result["answer"])
