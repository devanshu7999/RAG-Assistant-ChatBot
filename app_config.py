# # configurations and constants

# # SentenceTransformers embedding model
# EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# # Groq LLM
# GROQ_API_KEY = "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"
# GROQ_MODEL   = "llama-3.1-8b-instant"

# # ChromaDB
# CHROMA_PATH     = "data/chroma"
# COLLECTION_NAME = "local_rag_docs"

# # Chunking
# CHUNK_SIZE    = 1000
# CHUNK_OVERLAP = 200

# # Hybrid search
# HYBRID_TOP_K = 10   # candidates returned by hybrid search
# RERANK_TOP_K = 3    # chunks passed to the LLM
# ALPHA        = 0.5  # 0 = pure BM25 | 1 = pure dense | 0.5 = equal blend

# # PostgreSQL
# # DATABASE_URL = "postgresql://postgres:password@localhost:5433/rag_assistant"
# DATABASE_URL = "postgresql://neondb_owner:npg_cU6XIrbCLY5y@ep-aged-mode-ao7wcjps.c-2.ap-southeast-1.aws.neon.tech/neondb?sslmode=require"

# # Buffer Window Memory - number of recent messages to keep in-context
# BUFFER_WINDOW_SIZE = 10

# # Summary Memory - summarize after this many messages
# SUMMARY_THRESHOLD = 20

# # Knowledge Graph - Neo4j 
# NEO4J_URI      = "neo4j://localhost:7687"
# NEO4J_USER     = "neo4j"
# NEO4J_PASSWORD = "Devanshu@21"

# # Guardrails
# MAX_QUERY_LENGTH   = 300
# RELEVANCE_THRESHOLD = 0.15

# # Vector Store Memory (semantic long-term memory for logged-in users)
# VECTOR_MEMORY_TOP_K = 5   # past-conversation results to retrieve per query

# # Global Summary Memory (shared across all users and sessions)
# # Account Knowledge Graph is per-user (see memory_manager.py)
# MAX_GLOBAL_FACTS = 100    # max condensed knowledge facts in global summary memory


# configurations and constants
import os

# SentenceTransformers embedding model
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Groq LLM — read from Streamlit secrets / environment variable
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "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")
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

# PostgreSQL — read from Streamlit secrets / environment variable
DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://neondb_owner:npg_cU6XIrbCLY5y@ep-aged-mode-ao7wcjps.c-2.ap-southeast-1.aws.neon.tech/neondb?sslmode=require")

# Buffer Window Memory - number of recent messages to keep in-context
BUFFER_WINDOW_SIZE = 10

# Summary Memory - summarize after this many messages
SUMMARY_THRESHOLD = 20

# Knowledge Graph - Neo4j (optional — app works without it)
NEO4J_URI      = os.environ.get("NEO4J_URI", "neo4j://localhost:7687")
NEO4J_USER     = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "Devanshu@21")

# Guardrails
MAX_QUERY_LENGTH    = 300
RELEVANCE_THRESHOLD = 0.15

# Vector Store Memory (semantic long-term memory for logged-in users)
VECTOR_MEMORY_TOP_K = 5

# Global Summary Memory
MAX_GLOBAL_FACTS = 100