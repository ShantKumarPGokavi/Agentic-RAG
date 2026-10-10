from pathlib import Path

from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer


COLLECTION_NAME = "company_knowledge"
QDRANT_PATH = Path(__file__).parent / "qdrant_data"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

DOCUMENTS = [
    {
        "title": "AI acceptable use",
        "text": (
            "Employees may use company-approved AI tools to brainstorm, summarize "
            "public information, and draft routine work. Employees must review AI "
            "outputs for accuracy before using or sharing them."
        ),
    },
    {
        "title": "Confidential data and AI",
        "text": (
            "Do not enter customer personal data, passwords, private keys, "
            "unreleased financial results, or confidential source code into a "
            "public AI service. Use only company-approved systems for sensitive data."
        ),
    },
    {
        "title": "Human review of AI decisions",
        "text": (
            "A qualified employee must review AI-generated recommendations before "
            "they affect hiring, performance reviews, customer eligibility, safety, "
            "or other consequential decisions. AI must not be the sole decision-maker."
        ),
    },
    {
        "title": "AI output attribution",
        "text": (
            "Employees should disclose substantial AI assistance when required by "
            "the relevant team or publication guidelines. Verify facts and respect "
            "copyright and licensing requirements before reusing generated content."
        ),
    },
    {
        "title": "Remote work",
        "text": (
            "Employees may work remotely with manager approval. They should be "
            "available during their agreed team hours, protect company devices, and "
            "use the company VPN when accessing internal services on public networks."
        ),
    },
    {
        "title": "Time off",
        "text": (
            "Submit planned time-off requests to your manager in the HR system as "
            "early as possible. For illness or an urgent absence, notify your manager "
            "as soon as practical and record the absence in the HR system."
        ),
    },
    {
        "title": "Password and account security",
        "text": (
            "Use a unique password for each company account and store it in the "
            "approved password manager. Enable multi-factor authentication wherever "
            "available. Never share passwords or multi-factor authentication codes."
        ),
    },
    {
        "title": "Security incident reporting",
        "text": (
            "Report suspected phishing, lost devices, exposed credentials, or "
            "unexpected disclosure of company data to the IT security team promptly. "
            "Do not delete suspicious messages before the security team can review them."
        ),
    },
]


def main():
    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)
    vectors = model.encode([document["text"] for document in DOCUMENTS]).tolist()

    client = QdrantClient(path=str(QDRANT_PATH))
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=model.get_sentence_embedding_dimension(),
                distance=models.Distance.COSINE,
            ),
        )

    points = [
        models.PointStruct(
            id=index,
            vector=vectors[index],
            payload=document,
        )
        for index, document in enumerate(DOCUMENTS)
    ]
    client.upsert(collection_name=COLLECTION_NAME, points=points)
    client.close()
    print(f"Indexed {len(DOCUMENTS)} sample documents in {QDRANT_PATH}.")


if __name__ == "__main__":
    main()
