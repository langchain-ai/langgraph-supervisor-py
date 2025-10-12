from __future__ import annotations

import json
import logging
import os
from typing import Dict, List, Optional, Any

from whats_eat.app.env_loader import load_env
from neo4j import GraphDatabase
from openai import OpenAI

# Load environment variables including .env.json
load_env()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGTools:
    def __init__(self) -> None:
        # Neo4j connection
        self.neo4j_uri: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.neo4j_user: str = os.getenv("NEO4J_USER", "neo4j")
        self.neo4j_password: Optional[str] = os.getenv("NEO4J_PASSWORD")
        self.neo4j_driver = None

        # Pinecone connection
        self.pinecone_api_key: Optional[str] = os.getenv("PINECONE_API_KEY")
        self.pinecone_environment: Optional[str] = os.getenv("PINECONE_ENVIRONMENT")
        self.index_name: str = "places-index"

        # Pinecone runtime handles (lazy init) — supports new and legacy SDKs
        self._pinecone_client: Optional[object] = None
        self._pinecone_index: Optional[object] = None
        self._pinecone_new_sdk: bool = False

        # OpenAI client for embeddings
        self.openai_client = OpenAI()

    def connect_neo4j(self) -> None:
        """Connect to Neo4j using environment variables."""
        if not self.neo4j_password:
            raise ValueError("NEO4J_PASSWORD is required but missing in environment")
        try:
            self.neo4j_driver = GraphDatabase.driver(
                self.neo4j_uri, auth=(self.neo4j_user, self.neo4j_password)
            )
            logger.info("Connected to Neo4j")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise

    def connect_pinecone(self) -> None:
        """Initialize Pinecone, preferring the new SDK and falling back to legacy."""
        if not self.pinecone_api_key:
            raise ValueError("Missing PINECONE_API_KEY in environment")

        # Infer cloud/region from PINECONE_ENVIRONMENT (e.g., us-east-1-aws)
        cloud = "aws"
        region: Optional[str] = None
        if self.pinecone_environment:
            env_l = self.pinecone_environment.lower()
            if env_l.endswith("-aws"):
                cloud, region = "aws", env_l[:-4]
            elif env_l.endswith("-gcp"):
                cloud, region = "gcp", env_l[:-4]
            elif env_l.endswith("-azure"):
                cloud, region = "azure", env_l[:-6]
            else:
                region = env_l

        try:
            # Try new SDK first
            from pinecone import Pinecone as _Pinecone  # type: ignore
            from pinecone import ServerlessSpec as _ServerlessSpec  # type: ignore

            pc = _Pinecone(api_key=self.pinecone_api_key)
            existing = [ix.name for ix in pc.list_indexes()]
            if self.index_name not in existing:
                if not region:
                    region = "us-east-1"
                pc.create_index(
                    name=self.index_name,
                    dimension=1536,
                    metric="cosine",
                    spec=_ServerlessSpec(cloud=cloud, region=region),
                )
            self._pinecone_index = pc.Index(self.index_name)
            self._pinecone_client = pc
            self._pinecone_new_sdk = True
            logger.info("Initialized Pinecone (new SDK)")
        except Exception as new_err:
            # Fallback to legacy SDK if available
            try:
                import importlib

                pinecone_legacy = importlib.import_module("pinecone")
                pinecone_legacy.init(
                    api_key=self.pinecone_api_key, environment=self.pinecone_environment
                )
                if self.index_name not in pinecone_legacy.list_indexes():
                    pinecone_legacy.create_index(
                        name=self.index_name, dimension=1536, metric="cosine"
                    )
                self._pinecone_index = pinecone_legacy.Index(self.index_name)
                self._pinecone_client = pinecone_legacy
                self._pinecone_new_sdk = False
                logger.info("Initialized Pinecone (legacy SDK)")
            except Exception as legacy_err:
                msg = str(new_err) if new_err else ""
                if "pinecone-client" in msg or "renamed from `pinecone-client`" in msg:
                    raise RuntimeError(
                        "Pinecone SDK conflict: uninstall 'pinecone-client' and install 'pinecone' (new SDK)."
                    ) from new_err
                raise legacy_err

    def load_json_data(self, file_path: str) -> Any:
        """Read a JSON file and return parsed data (dict or list)."""
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _normalize_place(self, raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Normalize incoming place record to expected schema."""
        if not isinstance(raw, dict):
            return None

        if "place_id" in raw and "name" in raw:
            return raw

        # Try test format mapping
        pid = raw.get("id")
        dname = raw.get("displayName") or {}
        name = (dname.get("text") if isinstance(dname, dict) else None) or raw.get("name")
        addr = raw.get("formattedAddress") or raw.get("formatted_address")
        if pid and name:
            return {
                "place_id": pid,
                "name": name,
                "formatted_address": addr or "",
                "types": [],
                "rating": 0.0,
                "reviews": [],
            }
        return None

    def create_knowledge_graph(self, place_data: Dict[str, Any]) -> None:
        """Write place and (optionally) reviews into Neo4j."""
        if not self.neo4j_driver:
            raise ValueError("Neo4j is not connected; call connect_neo4j() first")

        with self.neo4j_driver.session() as session:
            session.run(
                """
                CREATE (p:Place {
                    place_id: $place_id,
                    name: $name,
                    address: $address,
                    rating: $rating,
                    types: $types
                })
                """,
                {
                    "place_id": place_data["place_id"],
                    "name": place_data["name"],
                    "address": place_data.get("formatted_address", ""),
                    "rating": place_data.get("rating", 0.0),
                    "types": place_data.get("types", []),
                },
            )

            if "reviews" in place_data:
                for r in place_data["reviews"]:
                    session.run(
                        """
                        MATCH (p:Place {place_id: $place_id})
                        CREATE (rv:Review {
                            author_name: $author_name,
                            rating: $rating,
                            text: $text,
                            time: $time
                        })
                        CREATE (rv)-[:REVIEWS]->(p)
                        """,
                        {
                            "place_id": place_data["place_id"],
                            "author_name": r.get("author_name"),
                            "rating": r.get("rating"),
                            "text": r.get("text"),
                            "time": r.get("time"),
                        },
                    )

        logger.info(f"KG upserted for place: {place_data.get('name')}")

    def create_embeddings(self, place_data: Dict[str, Any]) -> None:
        """Encode a textual representation and upsert into Pinecone."""
        parts: List[str] = [
            str(place_data.get("name", "")),
            str(place_data.get("formatted_address", "")),
            "Types: " + ", ".join(place_data.get("types", []) or []),
        ]
        if "reviews" in place_data and place_data["reviews"]:
            reviews_text = " ".join(str(rv.get("text", "")) for rv in place_data["reviews"]).strip()
            if reviews_text:
                parts.append("Reviews: " + reviews_text)

        text_representation = " ".join(p for p in parts if p).strip()
        response = self.openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=text_representation,
            encoding_format="float"
        )
        embedding = response.data[0].embedding

        if not self._pinecone_index:
            # Allow implicit init if caller forgot connect_pinecone
            self.connect_pinecone()

        if self._pinecone_new_sdk:
            self._pinecone_index.upsert(
                vectors=[
                    {
                        "id": place_data["place_id"],
                        "values": embedding,  # Already a list from OpenAI API
                        "metadata": {
                            "name": place_data.get("name"),
                            "address": place_data.get("formatted_address", ""),
                            "rating": place_data.get("rating", 0.0),
                        },
                    }
                ]
            )
        else:
            # Legacy SDK tuple style
            self._pinecone_index.upsert(
                vectors=[
                    (
                        place_data["place_id"],
                        embedding,  # Already a list from OpenAI API
                        {
                            "name": place_data.get("name"),
                            "address": place_data.get("formatted_address", ""),
                            "rating": place_data.get("rating", 0.0),
                        },
                    )
                ]
            )

        logger.info(f"Vector upserted for place: {place_data.get('name')}")

    def query_similar_places(self, query_text: str, top_k: int = 5) -> Any:
        """Encode the query and perform Pinecone vector search."""
        response = self.openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=query_text,
            encoding_format="float"
        )
        query_embedding = response.data[0].embedding
        
        if not self._pinecone_index:
            self.connect_pinecone()
        return self._pinecone_index.query(
            vector=query_embedding, top_k=top_k, include_metadata=True
        )