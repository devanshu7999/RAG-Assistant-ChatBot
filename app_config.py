# configurations and constants
import os

def _get_secret(key: str, fallback: str = "") -> str:
    """Read from Streamlit secrets first, then env vars, then fallback."""
    try:
        import streamlit as st
        val = st.secrets.get(key)
        if val:
            return str(val)
    except Exception:
        pass
    return os.environ.get(key, fallback)


# SentenceTransformers embedding model
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Groq LLM
GROQ_API_KEY = _get_secret("GROQ_API_KEY", "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")
GROQ_MODEL   = "llama-3.1-8b-instant"

# ChromaDB
CHROMA_PATH     = "data/chroma"
COLLECTION_NAME = "local_rag_docs"

# Chunking
CHUNK_SIZE    = 1000
CHUNK_OVERLAP = 200

# Hybrid search
HYBRID_TOP_K = 10
RERANK_TOP_K = 3
ALPHA        = 0.5

# PostgreSQL
DATABASE_URL = _get_secret("DATABASE_URL", "postgresql://neondb_owner:npg_1PtBgps0zxbE@ep-dry-sound-ao01gd1q.c-2.ap-southeast-1.aws.neon.tech/neondb?sslmode=require")

# Buffer Window Memory
BUFFER_WINDOW_SIZE = 10

# Summary Memory
SUMMARY_THRESHOLD = 20

# Knowledge Graph - Neo4j (optional)
NEO4J_URI      = _get_secret("NEO4J_URI", "neo4j+s://24f60106.databases.neo4j.io")
NEO4J_USER     = _get_secret("NEO4J_USER", "24f60106")
NEO4J_PASSWORD = _get_secret("NEO4J_PASSWORD", "d-IQoZ2PjzSerEWBFYI1nlEZr33nECDvVHCWtkoxR90")
# NEO4J_URI      = _get_secret("NEO4J_URI", "neo4j://localhost:7687")
# NEO4J_USER     = _get_secret("NEO4J_USER", "neo4j")
# NEO4J_PASSWORD = _get_secret("NEO4J_PASSWORD", "Devanshu@21")

# Guardrails
MAX_QUERY_LENGTH    = 300
RELEVANCE_THRESHOLD = 0.15

# Vector Store Memory
VECTOR_MEMORY_TOP_K = 5

# Global Summary Memory
MAX_GLOBAL_FACTS = 100