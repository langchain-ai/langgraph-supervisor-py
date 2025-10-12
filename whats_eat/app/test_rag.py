from whats_eat.agents.RAG_agent import RAGAgent
import pytest
import json

def print_divider(title):
    print("\n" + "="*50)
    print(title)
    print("="*50)

@pytest.fixture
def agent():
    """Create a fresh RAG agent for each test"""
    return RAGAgent()

def test_full_rag_pipeline(agent):
    """Test the complete RAG pipeline including Neo4j and Pinecone"""
    # Process test.json with full pipeline
    print_divider("Processing test.json")
    processed = agent.process_places_data("tests/test.json", dry_run=False)  # Enable full pipeline
    assert len(processed) > 0, "No places were processed"
    print(f"Processed {len(processed)} places")

    # Verify Neo4j data
    print_divider("Checking Neo4j data")
    with agent.rag_tools.neo4j_driver.session() as session:
        # Count places
        result = session.run("MATCH (p:Place) RETURN count(p) as count")
        count = result.single()["count"]
        print(f"Places in Neo4j: {count}")
        assert count > 0, "No places found in Neo4j"
        
        # Get all places
        result = session.run("MATCH (p:Place) RETURN p.name, p.place_id, p.address")
        places = [{
            "name": record["p.name"],
            "place_id": record["p.place_id"],
            "address": record["p.address"]
        } for record in result]
        print("\nNeo4j stored places:")
        print(json.dumps(places, indent=2))
        assert any("Eem" in place["name"] for place in places), \
            "Test restaurant not found in Neo4j"

    # Test vector search
    print_divider("Testing vector search")
    results = agent.query_similar_places("thai food in portland")
    matches = results.matches if hasattr(results, 'matches') else results.to_dict()['matches']
    
    print("\nVector search results:")
    print(json.dumps([{
        'id': match.id if hasattr(match, 'id') else match['id'],
        'score': match.score if hasattr(match, 'score') else match['score'],
        'metadata': match.metadata if hasattr(match, 'metadata') else match['metadata']
    } for match in matches], indent=2))
    
    assert len(matches) > 0, "No vector search results found"
    assert any("Thai" in str(match.metadata.get('name', '')) if hasattr(match, 'metadata')
              else match['metadata'].get('name', '') for match in matches), \
        "Expected Thai restaurant not found in vector search results"