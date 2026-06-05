import logging
from typing import Dict, Any, List
from backend.extractor import GeminiExtractor
from backend.graph_db import LegalGraphDB
from backend.vector_db import LegalVectorDB

logger = logging.getLogger(__name__)

class HybridRAGEngine:
    """
    Coordinates Vector Search, Graph Traversal, and Gemini to deliver
    highly accurate legal Q&A with deep dependency mapping.
    """
    def __init__(self, extractor: GeminiExtractor, graph_db: LegalGraphDB, vector_db: LegalVectorDB):
        self.extractor = extractor
        self.graph_db = graph_db
        self.vector_db = vector_db

    def process_document(self, doc_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Clears databases and processes a parsed list of document chunks:
        1. Generates embeddings and saves to ChromaDB.
        2. Uses Gemini to extract entities/relationships and stores in SQLite Graph.
        """
        # 1. Clear old data for a clean run
        self.vector_db.clear_db()
        self.graph_db.clear_db()

        total_chunks = len(doc_chunks)
        logger.info(f"Starting processing of {total_chunks} document chunks...")

        ids = []
        texts = []
        embeddings = []
        metadatas = []

        # List of triplets to insert after all nodes are processed
        all_triplets = []

        for idx, chunk in enumerate(doc_chunks):
            chunk_id = chunk["id"]
            title = chunk["title"]
            text = chunk["text"]
            page = chunk["page_num"]

            logger.info(f"[{idx+1}/{total_chunks}] Processing chunk: {title}")

            # A. Vector Database Preparation
            emb = self.extractor.get_embedding(text)
            ids.append(chunk_id)
            texts.append(text)
            embeddings.append(emb)
            metadatas.append({"title": title, "page_num": page})

            # B. Graph Database Node Addition
            self.graph_db.add_node(
                node_id=chunk_id,
                name=title,
                node_type="Clause",
                properties={"text": text, "page_num": page}
            )

            # C. Gemini Entity and Relationship Extraction
            extraction = self.extractor.extract_legal_triplets(title, text)
            
            # Add extracted entities (Definitions, Actors, Risks)
            for entity in extraction.get("entities", []):
                ent_name = entity.get("name")
                ent_type = entity.get("type", "Clause")
                if not ent_name:
                    continue
                
                # Format an ID for the entity
                ent_id = ent_name.lower().replace(" ", "_")
                
                # Check if it is a Risk and map properties
                properties = {}
                if ent_type == "Risk":
                    # Locate corresponding risk description from risks list
                    for r in extraction.get("risks", []):
                        if r.get("name") == ent_name:
                            properties = {
                                "severity": r.get("severity", "Medium"),
                                "text": r.get("description", "")
                            }
                            break
                            
                self.graph_db.add_node(ent_id, ent_name, ent_type, properties)

            # Queue relationships
            for rel in extraction.get("relationships", []):
                source = rel.get("source")
                relation = rel.get("relation", "REFERENCES")
                target = rel.get("target")
                if source and target:
                    all_triplets.append((source, relation, target))

        # 2. Bulk insert vectors into ChromaDB
        if ids:
            self.vector_db.add_clauses(ids, texts, embeddings, metadatas)

        # 3. Add all edges in Graph Database
        for src, rel, tgt in all_triplets:
            self.graph_db.add_edge(src, rel, tgt)

        logger.info("Document ingestion successfully completed.")
        return {"status": "success", "chunks_processed": total_chunks}

    def answer_query(self, query: str) -> Dict[str, Any]:
        """
        Runs Hybrid RAG:
        1. Vector search.
        2. Retrieve connected subgraphs for search results.
        3. Construct prompt & request answer from Gemini.
        """
        # 1. Embed query
        query_emb = self.extractor.get_embedding(query)
        
        # 2. Vector similarity search
        vector_results = self.vector_db.query_similarity(query_emb, limit=3)
        if not vector_results:
            return {
                "answer": "No terms document has been uploaded yet, or the uploaded document is empty. Please upload a T&C PDF first.",
                "sources": [],
                "subgraph": {"nodes": [], "edges": []}
            }

        # 3. Find clause titles to query in the Graph Database
        clause_titles = [doc["metadata"]["title"] for doc in vector_results]
        
        # 4. Traversal of Knowledge Graph to locate dependencies (Depth = 1 is standard)
        nodes_data, edges_data = self.graph_db.get_connected_subgraph(clause_titles, depth=1)

        # 5. Build Unified Context for Gemini
        context = "--- SEMANTICALLY RELEVANT CLAUSES (Vector Search) ---\n"
        for doc in vector_results:
            context += f"[{doc['metadata']['title']}] (Page {doc['metadata']['page_num']}):\n{doc['text']}\n\n"

        context += "--- CONNECTED CLAUSES & KNOWLEDGE (Graph DB) ---\n"
        seen_titles = {doc["metadata"]["title"] for doc in vector_results}
        for n in nodes_data:
            n_name = n["name"]
            n_type = n.get("type", "Clause")
            
            # Avoid repeating text of clauses we already included in the vector search
            if n_type == "Clause" and n_name not in seen_titles:
                text = n.get("text", "")
                if text:
                    context += f"[{n_name}] (Page {n.get('page_num', 1)}):\n{text}\n\n"
            elif n_type in ["Definition", "Risk", "Actor"]:
                severity_str = f" [Severity: {n.get('severity')}]" if n.get("severity") else ""
                desc = n.get("text", "") or n.get("description", "")
                context += f"- {n_type}: {n_name}{severity_str}"
                if desc:
                    context += f" - {desc}"
                context += "\n"
        context += "\n"

        if edges_data:
            context += "--- LEGAL RELATIONSHIPS & DEPENDENCIES ---\n"
            for edge in edges_data:
                context += f"- {edge['source']} --({edge['relation']})--> {edge['target']}\n"

        # 6. Request answer generation from Gemini
        answer = self.extractor.generate_answer(query, context)

        # 7. Format subgraph response for Vis.js UI highlighting
        subgraph_nodes = []
        name_to_id = {}
        for n in nodes_data:
            name_to_id[n["name"]] = n["id"]
            subgraph_nodes.append({
                "id": n["id"],
                "label": n["name"],
                "type": n.get("type", "Clause")
            })

        subgraph_edges = []
        for e in edges_data:
            src_id = name_to_id.get(e["source"], e["source"].lower().replace(" ", "_"))
            tgt_id = name_to_id.get(e["target"], e["target"].lower().replace(" ", "_"))
            subgraph_edges.append({
                "from": src_id,
                "to": tgt_id,
                "label": e["relation"]
            })

        return {
            "answer": answer,
            "sources": [
                {"title": doc["metadata"]["title"], "page": doc["metadata"]["page_num"], "text": doc["text"][:150] + "..."}
                for doc in vector_results
            ],
            "subgraph": {
                "nodes": subgraph_nodes,
                "edges": subgraph_edges
            }
        }
