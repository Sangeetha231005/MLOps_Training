import os
import chromadb
from typing import List, Dict, Any

class LegalVectorDB:
    """
    Manages the local vector database using ChromaDB to store and query legal clause embeddings.
    """
    def __init__(self, persist_dir: str = "../data/chromadb", collection_name: str = "legal_clauses"):
        # Ensure path is absolute/resolvable
        self.persist_dir = os.path.abspath(persist_dir)
        os.makedirs(self.persist_dir, exist_ok=True)
        
        # Initialize persistent Chroma client
        self.client = chromadb.PersistentClient(path=self.persist_dir)
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def clear_db(self):
        """
        Deletes the collection and recreates it to clear any old indices.
        """
        try:
            self.client.delete_collection(name=self.collection.name)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(name=self.collection.name)

    def add_clauses(self, ids: List[str], documents: List[str], embeddings: List[List[float]], metadatas: List[Dict[str, Any]]):
        """
        Adds text clauses, their Gemini embeddings, and metadata into the vector DB.
        """
        if not ids:
            return
        
        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )

    def query_similarity(self, query_embedding: List[float], limit: int = 4) -> List[Dict[str, Any]]:
        """
        Queries ChromaDB using the computed embedding of the user's question.
        Returns the top matching documents and their metadata.
        """
        # Ensure we have documents to query
        if self.collection.count() == 0:
            return []
            
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=limit
        )

        formatted_results = []
        if results and results["ids"] and results["ids"][0]:
            for i in range(len(results["ids"][0])):
                formatted_results.append({
                    "id": results["ids"][0][i],
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i] if "distances" in results and results["distances"] else 0.0
                })
        return formatted_results

if __name__ == "__main__":
    vdb = LegalVectorDB()
    vdb.clear_db()
    
    # Mock data test
    vdb.add_clauses(
        ids=["sec1", "sec2"],
        documents=["Section 1: The user agrees not to upload malware.", "Section 2: The company can terminate the account at will."],
        embeddings=[[0.1] * 768, [0.2] * 768],
        metadatas=[{"title": "Section 1", "page_num": 1}, {"title": "Section 2", "page_num": 2}]
    )
    
    # Query test
    res = vdb.query_similarity([0.1] * 768, limit=1)
    print("Vector Query Match:")
    print(res)
