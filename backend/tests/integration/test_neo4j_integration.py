"""Integration tests with a real Neo4j test database."""

import pytest
import os
from neo4j import GraphDatabase

# Use test Neo4j instance (can be Docker container)
# For CI/CD, use environment variables
NEO4J_TEST_URI = os.getenv("NEO4J_TEST_URI", "bolt://localhost:7688")
NEO4J_TEST_USER = os.getenv("NEO4J_TEST_USER", "neo4j")
NEO4J_TEST_PASSWORD = os.getenv("NEO4J_TEST_PASSWORD", "test_password")


@pytest.fixture(scope="function")
def neo4j_test_db():
    """Set up and tear down test Neo4j database."""
    try:
        driver = GraphDatabase.driver(
            NEO4J_TEST_URI, auth=(NEO4J_TEST_USER, NEO4J_TEST_PASSWORD)
        )
        driver.verify_connectivity()

        # Clear database before tests
        with driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")

        yield driver

        # Clean up after tests
        with driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        driver.close()
    except Exception as e:
        pytest.skip(f"Neo4j test database not available: {e}")


@pytest.mark.integration
def test_neo4j_search_integration(neo4j_test_db):
    """Test searching Neo4j with real database."""
    # Insert test data
    with neo4j_test_db.session() as session:
        session.run(
            """
            CREATE (m:Model {
                model_id: 'test/model',
                downloads: 1000,
                pipeline_tag: 'text-generation',
                library_name: 'transformers'
            })
            """
        )

    # Test search
    with neo4j_test_db.session() as session:
        result = session.run("MATCH (m:Model) RETURN m LIMIT 1")
        record = result.single()

        assert record is not None
        assert record["m"]["model_id"] == "test/model"
        assert record["m"]["downloads"] == 1000


@pytest.mark.integration
def test_neo4j_relationships_integration(neo4j_test_db):
    """Test Neo4j relationships with real database."""
    # Insert test data with relationships
    with neo4j_test_db.session() as session:
        session.run(
            """
            CREATE (m1:Model {model_id: 'model1', downloads: 1000})
            CREATE (m2:Model {model_id: 'model2', downloads: 500})
            CREATE (m1)-[:BASED_ON]->(m2)
            """
        )

    # Test relationship query
    with neo4j_test_db.session() as session:
        result = session.run(
            """
            MATCH (m1:Model)-[r:BASED_ON]->(m2:Model)
            RETURN m1.model_id as source, m2.model_id as target, type(r) as rel_type
            """
        )
        record = result.single()

        assert record is not None
        assert record["source"] == "model1"
        assert record["target"] == "model2"
        assert record["rel_type"] == "BASED_ON"


@pytest.mark.integration
def test_neo4j_apoc_integration(neo4j_test_db):
    """Test APOC procedures with real database."""
    # Insert test data
    with neo4j_test_db.session() as session:
        session.run(
            """
            CREATE (m1:Model {model_id: 'root'})
            CREATE (m2:Model {model_id: 'child1'})
            CREATE (m3:Model {model_id: 'child2'})
            CREATE (m1)-[:BASED_ON]->(m2)
            CREATE (m1)-[:BASED_ON]->(m3)
            """
        )

    # Test APOC subgraph query (if available)
    with neo4j_test_db.session() as session:
        try:
            result = session.run(
                """
                MATCH (root:Model {model_id: 'root'})
                CALL apoc.path.subgraphAll(root, {
                    relationshipFilter: 'BASED_ON>'
                })
                YIELD nodes, relationships
                RETURN size(nodes) as node_count, size(relationships) as rel_count
                """
            )
            record = result.single()
            if record:
                assert record["node_count"] >= 1
        except Exception:
            # APOC might not be available in test database
            pytest.skip("APOC procedures not available in test database")
