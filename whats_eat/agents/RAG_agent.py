from __future__ import annotations

import json
import logging
import sys
from typing import Dict, List, Any

from whats_eat.tools.RAG import RAGTools

logger = logging.getLogger(__name__)

class RAGAgent:
    def __init__(self) -> None:
        self.rag_tools = RAGTools()
        logger.info("Initialized RAG agent")
        logger.info(f"Neo4j URI: {self.rag_tools.neo4j_uri}")
        logger.info(f"Neo4j password present: {bool(self.rag_tools.neo4j_password)}")
        logger.info(f"Pinecone API key present: {bool(self.rag_tools.pinecone_api_key)}")
        logger.info(f"Pinecone environment: {self.rag_tools.pinecone_environment}")

    def process_places_data(self, json_file_path: str, dry_run: bool = False) -> List[Dict[str, Any]]:
        """End-to-end processing: load JSON, normalize, optionally connect, and upsert."""
        data = self.rag_tools.load_json_data(json_file_path)
        if isinstance(data, dict):
            items = data.get("results", [])
        elif isinstance(data, list):
            items = data
        else:
            items = []

        normalized = []
        for raw in items:
            doc = self.rag_tools._normalize_place(raw)
            if doc:
                normalized.append(doc)

        if not dry_run:
            self.rag_tools.connect_neo4j()
            self.rag_tools.connect_pinecone()
            for place in normalized:
                self.rag_tools.create_knowledge_graph(place)
                self.rag_tools.create_embeddings(place)
            logger.info("Finished processing places JSON: KG + embeddings upserted")
        else:
            logger.info(f"Dry run: would process {len(normalized)} places (no external connections)")

        return normalized

    def query_similar_places(self, query_text: str, top_k: int = 5) -> Any:
        """Encode the query and perform Pinecone vector search."""
        return self.rag_tools.query_similar_places(query_text, top_k)


if __name__ == "__main__":
    # CLI: allow running the agent with a JSON file path
    import argparse

    parser = argparse.ArgumentParser(
        prog="RAGAgent",
        description=(
            "Process Places-style JSON and store knowledge graph + embeddings; "
            "then you can query with query_similar_places()."
        ),
    )
    parser.add_argument(
        "json_file",
        nargs="?",
        help="Path to places JSON file to process (optional)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and normalize only; skip Neo4j/Pinecone connections",
    )

    args = parser.parse_args()
    agent = RAGAgent()

    if args.json_file:
        processed = agent.process_places_data(args.json_file, dry_run=args.dry_run)
        if args.dry_run:
            # Print a compact preview of normalized docs
            preview = [
                {"place_id": d.get("place_id"), "name": d.get("name"), "address": d.get("formatted_address")}
                for d in processed
            ]
            print(json.dumps(preview, ensure_ascii=False, indent=2))
    else:
        print(
            "No JSON file provided. You can run the module as:\n"
            "  py -m whats_eat.agents.RAG_agent path\\to\\places.json\n"
            "or:\n"
            "  py whats_eat\\agents\\RAG_agent.py path\\to\\places.json"
        )
        sys.exit(2)