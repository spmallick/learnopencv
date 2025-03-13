import xml.etree.ElementTree as ET

NAMESPACE = "{http://graphml.graphdrawing.org/xmlns}"

def xml_to_json(xml_path):
    """
    Parses a GraphML file with:
      - Nodes having keys: v_type, v_description, v_name
      - Edges having key: e_description

    Returns a dict:
    {
      "nodes": [
        {
          "id": "n0",
          "entity_type": "...",       # from v_type
          "description": "...",       # from v_description
          "source_id": "n0",          # same as node.id (optional)
          "name": "..."               # from v_name
        },
        ...
      ],
      "edges": [
        {
          "id": "e0",                # from edge @id or synthetic if missing
          "source": "n282",
          "target": "n308",
          "description": "...",      # from e_description
          "source_id": "e0"          # same as edge.id (optional)
        },
        ...
      ]
    }
    """

    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Find the <graph> element
    graph_elem = root.find(f".//{NAMESPACE}graph")

    nodes = []
    edges = []

    if graph_elem is None:
        return {"nodes": [], "edges": []}

    # --- Parse Nodes ---
    for node_elem in graph_elem.findall(f"{NAMESPACE}node"):
        node_id = node_elem.attrib.get("id")

        # Default structure
        node_data = {
            "id": node_id,
            "entity_type": None,
            "description": None,
            "source_id": node_id,  # Keep track of original ID if you want
            "name": None,
        }

        # Collect <data> children for this node
        for data_elem in node_elem.findall(f"{NAMESPACE}data"):
            key = data_elem.attrib.get("key", "")
            text_value = (data_elem.text or "").strip()

            if key == "v_type":
                node_data["entity_type"] = text_value
            elif key == "v_description":
                node_data["description"] = text_value
            elif key == "v_name":
                node_data["name"] = text_value

        nodes.append(node_data)

    # --- Parse Edges ---
    for edge_elem in graph_elem.findall(f"{NAMESPACE}edge"):
        edge_id = edge_elem.attrib.get("id")
        edge_source = edge_elem.attrib.get("source")
        edge_target = edge_elem.attrib.get("target")

        # If no explicit 'id' on the edge, generate one
        if not edge_id:
            edge_id = f"edge_{edge_source}_{edge_target}"

        edge_data = {
            "id": edge_id,
            "source": edge_source,
            "target": edge_target,
            "description": None,
            # Remove these if not needed:
            # "keywords": None,
            # "weight": 1.0,
            "source_id": edge_id,  # Keep the edge's own ID if you want
        }

        # Parse <data> elements for edges
        for data_elem in edge_elem.findall(f"{NAMESPACE}data"):
            key = data_elem.attrib.get("key", "")
            text_value = (data_elem.text or "").strip()

            if key == "e_description":
                edge_data["description"] = text_value
         
        edges.append(edge_data)

    return {"nodes": nodes, "edges": edges}
