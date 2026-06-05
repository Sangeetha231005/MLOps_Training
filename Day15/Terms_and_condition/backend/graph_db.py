import os
import sqlite3
import json
from typing import List, Dict, Any, Tuple
import networkx as nx

class LegalGraphDB:
    """
    Manages the Knowledge Graph storing legal clauses, actors, risks, and definitions.
    Implements local storage in SQLite with NetworkX graph traversal.
    """
    def __init__(self, db_path: str = "../data/truth_terms.db"):
        # Make sure directory exists
        db_dir = os.path.dirname(db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)
            
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """
        Initializes the SQLite tables for storing graph data.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create Nodes Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS nodes (
                id TEXT PRIMARY KEY,
                name TEXT UNIQUE,
                type TEXT,
                properties TEXT
            )
        """)
        
        # Create Edges Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS edges (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT,
                target TEXT,
                relation TEXT,
                properties TEXT,
                FOREIGN KEY(source) REFERENCES nodes(name),
                FOREIGN KEY(target) REFERENCES nodes(name),
                UNIQUE(source, target, relation)
            )
        """)
        
        conn.commit()
        conn.close()

    def clear_db(self):
        """
        Wipes the database tables for a fresh ingest.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM edges")
        cursor.execute("DELETE FROM nodes")
        conn.commit()
        conn.close()

    def add_node(self, node_id: str, name: str, node_type: str, properties: Dict[str, Any] = None):
        """
        Adds a node to the graph database.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        props_str = json.dumps(properties or {})
        try:
            cursor.execute(
                "INSERT INTO nodes (id, name, type, properties) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(name) DO UPDATE SET type=excluded.type, properties=excluded.properties",
                (node_id, name, node_type, props_str)
            )
            conn.commit()
        except Exception as e:
            print(f"Error adding node {name}: {e}")
        finally:
            conn.close()

    def add_edge(self, source_name: str, relation: str, target_name: str, properties: Dict[str, Any] = None):
        """
        Adds an edge/relationship between two nodes. Ensures referenced nodes exist.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Helper to ensure nodes exist
        def ensure_node_exists(node_name: str):
            cursor.execute("SELECT 1 FROM nodes WHERE name = ?", (node_name,))
            if not cursor.fetchone():
                # Default type to 'Clause' or 'Actor' based on heuristics
                # Auto-generate a node ID
                node_id = node_name.lower().replace(" ", "_")
                node_type = "Clause" if ("section" in node_id or "clause" in node_id) else "Actor"
                cursor.execute(
                    "INSERT OR IGNORE INTO nodes (id, name, type, properties) VALUES (?, ?, ?, ?)",
                    (node_id, node_name, node_type, json.dumps({}))
                )

        props_str = json.dumps(properties or {})
        try:
            ensure_node_exists(source_name)
            ensure_node_exists(target_name)
            
            cursor.execute(
                "INSERT INTO edges (source, target, relation, properties) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(source, target, relation) DO UPDATE SET properties=excluded.properties",
                (source_name, relation, target_name, props_str)
            )
            conn.commit()
        except Exception as e:
            print(f"Error adding edge {source_name} -> {relation} -> {target_name}: {e}")
        finally:
            conn.close()

    def build_networkx_graph(self) -> nx.DiGraph:
        """
        Loads SQLite data into a NetworkX Directed Graph.
        """
        G = nx.DiGraph()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Load nodes
        cursor.execute("SELECT id, name, type, properties FROM nodes")
        for row in cursor.fetchall():
            node_id, name, type_, props = row
            G.add_node(name, id=node_id, type=type_, **json.loads(props))
            
        # Load edges
        cursor.execute("SELECT source, target, relation, properties FROM edges")
        for row in cursor.fetchall():
            src, tgt, rel, props = row
            G.add_edge(src, tgt, relation=rel, **json.loads(props))
            
        conn.close()
        return G

    def get_connected_subgraph(self, starting_nodes: List[str], depth: int = 1) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Finds a subgraph of connected clauses/entities centered around a starting set of nodes.
        Used to pull context during retrieval.
        """
        G = self.build_networkx_graph()
        subgraph_nodes = set()
        
        for start_node in starting_nodes:
            if not G.has_node(start_node):
                # Search for approximate matches
                matches = [n for n in G.nodes if start_node.lower() in n.lower() or n.lower() in start_node.lower()]
                curr_starts = matches
            else:
                curr_starts = [start_node]
                
            for s in curr_starts:
                subgraph_nodes.add(s)
                # Expand outward up to depth
                current_layer = {s}
                for _ in range(depth):
                    next_layer = set()
                    for node in current_layer:
                        # Add outgoing connections
                        for succ in G.successors(node):
                            next_layer.add(succ)
                        # Add incoming connections
                        for pred in G.predecessors(node):
                            next_layer.add(pred)
                    subgraph_nodes.update(next_layer)
                    current_layer = next_layer

        # Build list of node and edge dictionaries to return
        nodes_list = []
        for n in subgraph_nodes:
            node_data = G.nodes[n].copy()
            node_data["name"] = n
            nodes_list.append(node_data)
            
        edges_list = []
        for u, v, data in G.edges(subgraph_nodes, data=True):
            if u in subgraph_nodes and v in subgraph_nodes:
                edge_data = data.copy()
                edge_data["source"] = u
                edge_data["target"] = v
                edges_list.append(edge_data)
                
        return nodes_list, edges_list

    def get_all_graph_data(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Returns all nodes and edges. Perfect for Vis.js visualization.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT id, name, type, properties FROM nodes")
        nodes = []
        name_to_id = {}
        for row in cursor.fetchall():
            node_id, name, type_, props = row
            name_to_id[name] = node_id
            props_dict = json.loads(props)
            nodes.append({
                "id": node_id,
                "label": name,
                "type": type_,
                "page": props_dict.get("page_num", 1),
                "text": props_dict.get("text", "")
            })
            
        cursor.execute("SELECT source, target, relation FROM edges")
        edges = []
        for row in cursor.fetchall():
            src, tgt, rel = row
            # Locate corresponding IDs for Vis.js mapping
            src_id = name_to_id.get(src, src.lower().replace(" ", "_"))
            tgt_id = name_to_id.get(tgt, tgt.lower().replace(" ", "_"))
            edges.append({
                "from": src_id,
                "to": tgt_id,
                "label": rel
            })
            
        conn.close()
        return {"nodes": nodes, "edges": edges}

if __name__ == "__main__":
    db = LegalGraphDB()
    db.clear_db()
    
    # Test insertions
    db.add_node("section_5", "Section 5", "Clause", {"text": "Liabilities arise here.", "page_num": 2})
    db.add_node("section_10", "Section 10", "Clause", {"text": "Liability limitations apply.", "page_num": 3})
    db.add_node("user_data", "User Data", "Definition", {"text": "Any data uploaded.", "page_num": 1})
    
    db.add_edge("Section 5", "REFERENCES", "Section 10")
    db.add_edge("Section 5", "CONTAINS", "User Data")
    
    print("Full graph visual data:")
    print(json.dumps(db.get_all_graph_data(), indent=2))
    
    print("\nTraversing connected subgraph:")
    nodes, edges = db.get_connected_subgraph(["Section 5"], depth=1)
    print("Nodes:", [n["name"] for n in nodes])
    print("Edges:", [(e["source"], e["relation"], e["target"]) for e in edges])
