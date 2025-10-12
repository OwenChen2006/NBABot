import os

DB_DSN = os.getenv("DB_DSN", "postgresql://nba:nba@db:5432/nba")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://ollama:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2:3b")
