# configurations and constants

# SentenceTransformers embedding model
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Groq LLM
GROQ_API_KEY = "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"
GROQ_MODEL   = "llama-3.1-8b-instant"

# ChromaDB
CHROMA_PATH     = "data/chroma"
COLLECTION_NAME = "local_rag_docs"

# Chunking
CHUNK_SIZE    = 1000
CHUNK_OVERLAP = 200

# Hybrid search
HYBRID_TOP_K = 10   # candidates returned by hybrid search
RERANK_TOP_K = 3    # chunks passed to the LLM
ALPHA        = 0.5  # 0 = pure BM25 | 1 = pure dense | 0.5 = equal blend

# Memory
MEMORI_API_KEY = "IzOzBqku7LstfoWwhiTqhZRpg5N1QgrEK6Slj57GU9DS45Toqw1QrQDBB3F_LbmecYbJ_3tDAraAqcIwoaa5yUb5IK8qHuviV_E-n9URM1KkxCZ4ukNidwRm6aQ7rggNjBuvGkSqqmFeShj2PMMAnCka7FmHw7ERY2Xl02j4YXc"

# PostgreSQL
DATABASE_URL = "postgresql://postgres:password@localhost:5433/rag_assistant"

# Buffer Window Memory - number of recent messages to keep in-context
BUFFER_WINDOW_SIZE = 10

# Summary Memory - summarize after this many messages
SUMMARY_THRESHOLD = 20

# Knowledge Graph - Neo4j 
NEO4J_URI      = "neo4j://localhost:7687"
NEO4J_USER     = "neo4j"
NEO4J_PASSWORD = "Devanshu@21"

# Guardrails
MAX_QUERY_LENGTH   = 300
RELEVANCE_THRESHOLD = 0.15

# Vector Store Memory (semantic long-term memory for logged-in users)
VECTOR_MEMORY_TOP_K = 5   # past-conversation results to retrieve per query

# Global Memory (shared across all users and sessions)
MAX_GLOBAL_FACTS = 100    # max condensed knowledge facts in global summary memory
