import os
from typing import Final

# Environment variables with defaults
GOOGLE_MAPS_API_KEY: Final[str] = os.getenv('GOOGLE_MAPS_API_KEY', '')
YOUTUBE_API_KEY: Final[str] = os.getenv('YOUTUBE_API_KEY', '')
NEO4J_URI: Final[str] = os.getenv('NEO4J_URI', 'neo4j://127.0.0.1:7687')
NEO4J_USER: Final[str] = os.getenv('NEO4J_USER', 'neo4j')
NEO4J_PASSWORD: Final[str] = os.getenv('NEO4J_PASSWORD', '')
PINECONE_API_KEY: Final[str] = os.getenv('PINECONE_API_KEY', '')
PINECONE_ENVIRONMENT: Final[str] = os.getenv('PINECONE_ENVIRONMENT', 'us-east-1-aws')

# API retry strategy
MAX_RETRIES: Final[int] = 2          # Max exponential backoff retries
TIMEOUT_SEC: Final[int] = 20         # Default API timeout

# Places API tuning
DEFAULT_RADIUS_M: Final[int] = 2000  # 2km default search radius
MAX_PLACES: Final[int] = 25         # Max places to return
BACKOFF_BASE_SEC: Final[int] = 2    # Base for exponential backoff

# Storage
PINECONE_INDEX: Final[str] = 'places-index'
NEO4J_DATABASE: Final[str] = 'neo4j'

# Model parameters
EMBEDDING_MODEL: Final[str] = 'text-embedding-3-small'
EMBEDDING_DIM: Final[int] = 1536

# Rate limiting (requests/sec)
YOUTUBE_RATE_LIMIT: Final[float] = 1.0  # Max 1 request per second
PLACES_RATE_LIMIT: Final[float] = 10.0  # Max 10 requests per second