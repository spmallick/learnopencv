import os
import json
from neo4j_util import xml_to_json  # <= your custom import; adjust accordingly
from neo4j import GraphDatabase

# Constants
BATCH_SIZE_NODES = 500
BATCH_SIZE_EDGES = 100

# Neo4j connection credentials
NEO4J_URI = "neo4j+s://25ef5ce8.databases.neo4j.io"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "b5plqPqdvTRlEM4NCl_5Gs7T7JXTmGzx"   # Your neo4j instance password.


def convert_xml_to_json(xml_path, output_path):
    """Converts XML file to JSON and saves the output."""
    if not os.path.exists(xml_path):
        print(f"Error: File not found - {xml_path}")
        return None

    json_data = xml_to_json(xml_path)
    if json_data:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
        print(f"JSON file created: {output_path}")
        return json_data
    else:
        print("Failed to create JSON data")
        return None

def process_in_batches(tx, query, data, batch_size):
    """Process data in batches and execute the given query."""
    for i in range(0, len(data), batch_size):
        batch = data[i : i + batch_size]
        if "UNWIND $nodes" in query:
            tx.run(query, {"nodes": batch})
        else:
            tx.run(query, {"edges": batch})

def main():
    # Paths
    xml_file = "neo4j_graph/mimic430__graph_chunk_entity_relation.graphml"
    json_file = "neo4j_graph/json_mimic430__graph_chunk_entity_relation.json"
    
    # 1) Convert XML to JSON
    json_data = convert_xml_to_json(xml_file, json_file)
    if json_data is None:
        return

    # 2) Load nodes and edges from that JSON
    nodes = json_data.get("nodes", [])
    edges = json_data.get("edges", [])

    # 3) Define the Cypher queries

    # Create nodes (merges on id)
    create_nodes_query = """
    UNWIND $nodes AS node
    MERGE (e:Entity {id: node.id})
    SET e.entity_type = node.entity_type,
        e.description = node.description,
        e.source_id = node.source_id,
        e.displayName = node.id
    REMOVE e:Entity
    WITH e, node
    CALL apoc.create.addLabels(e, [node.id]) YIELD node AS labeledNode
    RETURN count(*)
    """

    # Create edges – uses edge.description as the relationship type
    # If edge.description is empty or None, default to 'IS'
    # *****************
    # /* comments 
    # -----------------
    create_edges_query = """
    /* Unwinding the edges from the batch */
    UNWIND $edges AS edge
    
    /* Matching the source and target nodes based on their IDs */
    MATCH (source {id: edge.source})
    MATCH (target {id: edge.target})

    /* Determining the relationship type using edge description */
    WITH source, target, edge,
         CASE
           WHEN edge.description IS NULL OR TRIM(edge.description) = '' THEN 'IS'
           ELSE TRIM(edge.description)
         END AS relType
    
    /* Creating the relationship between source and target nodes */
    CALL apoc.create.relationship(
      source, 
      relType, 
      {
        description: edge.description,
        source_id: edge.source_id,
        target_id: edge.target
      },   
      target    
    ) YIELD rel
    RETURN count(*)
    """


    # Optional: Set displayName and labels based on entity_type
    set_displayname_and_labels_query = """
    MATCH (n)
    SET n.displayName = n.id
    WITH n
    CALL apoc.create.setLabels(n, [n.entity_type]) YIELD node
    RETURN count(*)
    """

    # 4) Connect to Neo4j and run the queries in batches
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    try:
        with driver.session() as session:
            # Insert nodes in batches
            session.execute_write(
                process_in_batches, create_nodes_query, nodes, BATCH_SIZE_NODES
            )

            # Insert edges in batches
            session.execute_write(
                process_in_batches, create_edges_query, edges, BATCH_SIZE_EDGES
            )

            # Set displayName and labels
            session.run(set_displayname_and_labels_query)

    except Exception as e:
        print(f"Error occurred: {e}")

    finally:
        driver.close()

if __name__ == "__main__":
    main()
